"""
Module: train.py

This module provides functions to train and evaluate a video classification model.
It includes the main training loop that tracks loss and accuracy over epochs,
updates the best performing model based on validation accuracy, and provides helper
functions to compute the learning rate, batch loss, and epoch loss.

Functions:
    - train: Runs the training and validation loops over a specified number of epochs.
    - get_learning_rate: Retrieves the current learning rate from an optimizer.
    - batch_correct_preds: Computes the number of correct predictions in a mini-batch.
    - get_batch_loss: Computes the loss for a mini-batch and performs backpropagation.
    - get_epoch_loss: Computes the average loss and accuracy over an epoch.
"""

import os
import copy
from tqdm import tqdm
import torch
import wandb

from evaluate import evaluate, model_forward

def train(dataloaders, model, criterion, optimizer, scheduler, device, optim_model_wts_dir,
          n_epochs=30, class_names=None):
    """
    Train and validate the model over a given number of epochs.

    This function performs the training loop for a video classification model.
    It iterates through the specified number of epochs, updates model weights
    using backpropagation, and evaluates model performance on a validation set.
    The model with the best validation accuracy is saved to disk as best.pt, the weights after the
    last epoch as last.pt. At the end the best weights are reloaded and, when a Weights & Biases
    run is active, their confusion matrices on the validation and test sets are logged.

    Args:
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders, optionally 'test'
                            (evaluated every epoch for logging only; never used for model
                            selection).
        model (torch.nn.Module): The video classification model to be trained.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler: ReduceLROnPlateau (stepped
            with the validation loss) or an epoch-wise scheduler such as the cosine schedule.
        device (torch.device): Device (CPU or GPU) on which to perform training.
        optim_model_wts_dir (str): Directory to save the best model weights.
        n_epochs (int, optional): Number of training epochs. Default is 30.
        class_names (list, optional): Class names in label order, for the confusion matrices.

    Returns:
        tuple: (model, loss_hist, acc_hist)
            - model (torch.nn.Module): The trained model loaded with the best validation weights.
            - loss_hist (dict): Lists of training and validation losses for each epoch.
            - acc_hist (dict): Lists of training and validation accuracies for each epoch.
    """
    loss_hist = {'train': [], 'val': []}
    acc_hist = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(n_epochs):
        current_lr = get_learning_rate(optimizer)
        print(f'Epoch {epoch + 1}/{n_epochs}; Current learning rate {current_lr}')
        metrics = {'epoch': epoch + 1, 'lr': current_lr}

        # Training phase
        model.train()
        metrics['train/loss'], metrics['train/acc'] = get_epoch_loss(
            model, criterion, dataloaders['train'], device, optimizer)

        # Validation phase, then the test phase (logging only, never used for selection)
        model.eval()
        with torch.no_grad():
            metrics['val/loss'], metrics['val/acc'] = get_epoch_loss(
                model, criterion, dataloaders['val'], device)
            if 'test' in dataloaders:
                metrics['test/loss'], metrics['test/acc'] = get_epoch_loss(
                    model, criterion, dataloaders['test'], device)
        if metrics['val/acc'] > best_val_acc:
            best_val_acc = metrics['val/acc']
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(optim_model_wts_dir, 'best.pt'))
            print(f'Best model weights are updated at epoch {epoch + 1}!')
        torch.save(model.state_dict(), os.path.join(optim_model_wts_dir, 'last.pt'))
        loss_hist['train'].append(metrics['train/loss'])
        acc_hist['train'].append(metrics['train/acc'])
        loss_hist['val'].append(metrics['val/loss'])
        acc_hist['val'].append(metrics['val/acc'])

        # One Weights & Biases record per epoch (skipped when no run was started, e.g. in tests)
        if wandb.run is not None:
            wandb.log(metrics)

        # Update the learning rate. The hand-over's plateau scheduler halves it when the
        # validation loss stalls and then restarts from the best weights; a cosine schedule
        # changes it every epoch and simply continues.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metrics['val/loss'])
            if current_lr != get_learning_rate(optimizer):
                print('Loading best model weights!')
                model.load_state_dict(best_model_wts)
        else:
            scheduler.step()

        print(f"train loss: {metrics['train/loss']:.6f}, val loss: {metrics['val/loss']:.6f}, "
              f"accuracy: {100 * metrics['val/acc']:.2f}")
        if 'test/acc' in metrics:
            print(f"test loss: {metrics['test/loss']:.6f}, "
                  f"test accuracy: {100 * metrics['test/acc']:.2f}")
        print('-' * 60)
        print()

    # Load the best model weights before returning the model
    model.load_state_dict(best_model_wts)
    if wandb.run is not None:
        log_confusion_matrices(model, dataloaders, device, class_names)
    return model, loss_hist, acc_hist


def log_confusion_matrices(model, dataloaders, device, class_names):
    """
    Log a Weights & Biases confusion matrix of the current weights for the val and test sets.

    Args:
        model (torch.nn.Module): The model to evaluate (the best-validation weights).
        dataloaders (dict): DataLoaders by name; 'val' and, if present, 'test' are used.
        device (torch.device): Device on which to run the model.
        class_names (list): Class names in label order (None for numeric labels).
    """
    for name in ('val', 'test'):
        if name in dataloaders:
            targets, outputs, _ = evaluate(model, dataloaders[name], device)
            wandb.log({f'{name}/confusion_matrix': wandb.plot.confusion_matrix(
                y_true=targets, preds=outputs, class_names=class_names)})

def get_learning_rate(optimizer):
    """
    Retrieve the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer from which to get the learning rate.

    Returns:
        float: The current learning rate of the first parameter group (the LSTM and classifier
            when LRCN.parameter_groups is used; the backbone group may have a smaller one).
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def batch_correct_preds(output, target):
    """
    Compute the number of correct predictions for a mini-batch.

    Args:
        output (torch.Tensor): Model outputs (logits) with shape (batch_size, num_classes).
        target (torch.Tensor): True labels with shape (batch_size).

    Returns:
        int: Number of correct predictions in the mini-batch.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct_preds = pred.eq(target.view_as(pred)).sum().item()
    return correct_preds

def get_batch_loss(criterion, output, target, optimizer=None):
    """
    Compute the loss for a mini-batch and perform backpropagation (if optimizer is provided).

    Args:
        criterion (callable): Loss function.
        output (torch.Tensor): Model outputs for the mini-batch.
        target (torch.Tensor): True labels for the mini-batch.
        optimizer (torch.optim.Optimizer, optional): Optimizer to update model weights. If None,
            no backpropagation is performed.

    Returns:
        tuple: (loss_value, n_batch_correct_preds)
            - loss_value (float): Loss value for the mini-batch.
            - n_batch_correct_preds (int): Number of correct predictions in the mini-batch.
    """
    loss = criterion(output, target)
    with torch.no_grad():
        n_batch_correct_preds = batch_correct_preds(output, target)
    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), n_batch_correct_preds

def get_epoch_loss(model, criterion, dataloader, device, optimizer=None):
    """
    Compute the average loss and overall accuracy for an epoch.

    Iterates over the entire DataLoader, computes loss and accuracy for each mini-batch,
    and aggregates the results over the epoch.

    Args:
        model (torch.nn.Module): The video classification model.
        criterion (callable): Loss function.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device (CPU or GPU) on which to perform computations.
        optimizer (torch.optim.Optimizer, optional): If provided, used to update model weights
            during training.

    Returns:
        tuple: (loss, accuracy)
            - loss (float): Average loss over the epoch.
            - accuracy (float): Overall accuracy over the epoch.
    """
    running_loss, running_total_correct_preds = 0.0, 0.0
    len_dataset = len(dataloader.dataset)

    for x_batch, y_batch in tqdm(dataloader):
        # Forward under bf16 autocast on the GPU (see evaluate.model_forward); the loss, the
        # accuracy and the optimizer step use the float32 logits it returns.
        output, y_batch = model_forward(model, x_batch, device), y_batch.to(device)
        batch_loss, n_batch_correct_preds = get_batch_loss(criterion, output, y_batch, optimizer)

        running_loss += batch_loss
        running_total_correct_preds += n_batch_correct_preds

    loss = running_loss / float(len_dataset)
    accuracy = running_total_correct_preds / float(len_dataset)
    return loss, accuracy

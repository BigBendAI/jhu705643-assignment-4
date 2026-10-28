"""
Module: evaluate.py

This module provides functions for evaluating a video classification model on held-out data.
(It was test.py in the hand-over; that name clashes with pytest's test files and with Python's
own test package.)
It includes functions to compute predictions and accuracy, generate a detailed classification report,
and compute a multilabel confusion matrix for all classes.

Functions:
    - evaluate: Evaluates the model on a DataLoader and returns the ground truth labels,
      predicted labels, and overall accuracy.
    - get_test_report: Generates a classification report using scikit-learn's classification_report.
    - get_confusion_matrix: Computes a multilabel confusion matrix for each class.
"""

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix

def model_forward(model, x_batch, device):
    """
    Run the model on one batch under bf16 autocast on the GPU and return float32 logits.

    Under autocast the convolutions and matrix products run in bfloat16, which is about 1.5x
    faster on an A100; bfloat16 keeps float32's exponent range, so no loss scaling is needed.
    The caller works with the float32 copy of the logits. On the CPU autocast stays off. Used by
    training and evaluation so both compute the same thing.

    Args:
        model (torch.nn.Module): The model.
        x_batch (Tensor): A batch of clips.
        device (torch.device): Device the batch is moved to.

    Returns:
        Tensor: float32 logits of shape (batch_size, n_classes).
    """
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == 'cuda'):
        output = model(x_batch.to(device))
    return output.float()


def evaluate(model, dataloader, device):
    """
    Evaluate the model on a dataset and compute overall accuracy.
    
    This function sets the model to evaluation mode and processes the test data
    from the provided DataLoader. It computes predictions for each batch, counts the number
    of correct predictions, and accumulates the true and predicted labels.
    
    Args:
        model (torch.nn.Module): The trained video classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.
    
    Returns:
        tuple: (targets, outputs, accuracy)
            - targets (list): Ground truth labels for all samples.
            - outputs (list): Predicted labels for all samples.
            - accuracy (float): Overall accuracy computed as the ratio of correct predictions
                                to the total number of samples.
    """
    model.eval()
    with torch.no_grad():
        total_correct_preds = 0.0
        len_dataset = len(dataloader.dataset)
        targets, outputs = [], []
        for x_batch, y_batch in tqdm(dataloader):
            output, y_batch = model_forward(model, x_batch, device), y_batch.to(device)
            pred = output.argmax(dim=1, keepdim=True)
            correct_preds = pred.eq(y_batch.view_as(pred)).sum().item()
            total_correct_preds += correct_preds
            outputs.extend(pred.view(-1).detach().cpu().numpy().tolist())
            targets.extend(y_batch.detach().cpu().numpy().tolist())
        
        accuracy = total_correct_preds / float(len_dataset)
    
    return targets, outputs, accuracy

def get_test_report(target, output, target_names):
    """
    Generate a detailed classification report based on test results.
    
    This function uses scikit-learn's classification_report to produce a dictionary
    containing precision, recall, F1-score, and support for each class.
    
    Args:
        target (list): Ground truth labels.
        output (list): Predicted labels.
        target_names (list): List of class names corresponding to the labels.
    
    Returns:
        dict: A classification report as a dictionary.
    """
    return classification_report(target, output, output_dict=True, target_names=target_names)

def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    """
    Compute a multilabel confusion matrix for each class.
    
    This function converts numeric labels to their corresponding class names using the provided
    labels_dict, then computes a multilabel confusion matrix for each class using scikit-learn.
    
    Args:
        targets (list): Ground truth numeric labels.
        outputs (list): Predicted numeric labels.
        labels_dict (dict): Dictionary mapping class names to numeric labels.
        all_cats (list): List of all class names.
    
    Returns:
        dict: A dictionary where keys are class names and values are the corresponding confusion matrices.
    """
    # Create an inverse mapping from numeric label to class name
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = multilabel_confusion_matrix(target_cats, output_cats, labels=all_cats)
    return {label: mat for label, mat in zip(all_cats, confusion_mat)}

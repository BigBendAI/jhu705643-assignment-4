"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature
extraction from individual frames with an LSTM to capture temporal dynamics across frames.
An additional fully-connected layer is used to output the final class predictions.

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace the fully-connected
              layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization, and a final
          fully-connected layer to produce class logits.
    VideoCNN: The 3D CNN the hand-over's '3dcnn' model type was written for (R3D-18 pretrained on
          Kinetics-400), used as a third stream next to the RGB and flow LRCNs.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import video

# ResNet constructors and their ImageNet weights. torchvision removed the ``pretrained=`` argument;
# the current API is ``weights=``.
BACKBONES = {
    'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
    'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT),
    'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
    'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT),
    'resnet152': (models.resnet152, models.ResNet152_Weights.DEFAULT),
}

# 3D CNNs pretrained on the Kinetics-400 video dataset (torchvision) for --model_type 3dcnn.
VIDEO_BACKBONES = {
    'r3d_18': (video.r3d_18, video.R3D_18_Weights.KINETICS400_V1),
    'r2plus1d_18': (video.r2plus1d_18, video.R2Plus1D_18_Weights.KINETICS400_V1),
}


class Identity(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.
    
    This module is used to replace the fully-connected (fc) layer in the ResNet backbone,
    effectively making the backbone output the raw features before classification.
    
    Example:
        >>> identity = Identity()
        >>> output = identity(input_tensor)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass that returns the input as is.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: The same tensor x.
        """
        return x

class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.
    
    This model uses a ResNet backbone as a 2D CNN to extract spatial features from each video frame.
    An LSTM network is then used to model the temporal dynamics across the sequence of frame features.
    Dropout is applied before the final fully-connected layer that produces class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, uses a ResNet model pretrained on ImageNet. Default is True.
        cnn_model (str, optional): Specifies the ResNet variant to use as the backbone.
                                   Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.
                                   Default is 'resnet34'.
        temporal_pooling (str, optional): 'last' classifies the LSTM output of the last frame
            (the hand-over); 'late_fusion' classifies every frame's output and averages the
            logits over time, as the LRCN paper does. Default is 'last'.
        freeze_bn (bool, optional): Keep the backbone's BatchNorm layers in eval mode while
            training, so they use the ImageNet running statistics instead of re-estimating them
            from a few clips per batch. Default is False.
        freeze_stages (int, optional): Number of leading backbone stages whose weights stay
            fixed: 1 = stem + layer1, 2 = + layer2, ..., 4 = the whole backbone. Early stages
            hold generic edge and texture filters that need no adaptation. Default is 0.
        feature_dropout (float, optional): Dropout on every frame's CNN feature vector before
            the LSTM. Default is 0.0.
        bidirectional (bool, optional): Read the frame sequence forwards and backwards with two
            LSTMs whose outputs are concatenated, so each time step's output knows the whole
            clip. Default is False.

    Raises:
        ValueError: If the specified cnn_model or temporal_pooling is not supported.
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True,
                 cnn_model='resnet34', temporal_pooling='last', freeze_bn=False, freeze_stages=0,
                 feature_dropout=0.0, bidirectional=False):
        super(LRCN, self).__init__()
        if temporal_pooling not in ('last', 'late_fusion'):
            raise ValueError("temporal_pooling must be 'last' or 'late_fusion'")
        self.temporal_pooling = temporal_pooling
        self.freeze_bn = freeze_bn

        # Set up the ResNet backbone as a 2D CNN feature extractor. (The original built a resnet34
        # when 'resnet152' was requested.)
        if cnn_model not in BACKBONES:
            raise ValueError('The input CNN backbone is not supported, please choose a valid ResNet variant.')
        constructor, weights = BACKBONES[cnn_model]
        base_cnn = constructor(weights=weights if pretrained else None)

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features
        
        # Replace the original fc layer with an identity mapping so that raw features are returned.
        base_cnn.fc = Identity()
        # channels-last (NHWC) is the layout the GPU convolution kernels are fastest in; the weights
        # are stored that way once, and forward() hands the frames over in the same layout.
        self.base_model = base_cnn.to(memory_format=torch.channels_last)
        self.freeze_backbone_stages(freeze_stages)
        self.feature_dropout = nn.Dropout(feature_dropout)

        # Define the LSTM to process the sequence of frame features. batch_first=True makes it
        # read its input as (batch, time, features), which is what forward() builds. Without it
        # the LSTM expects (time, batch, features) and reads the batch axis as time.
        # With more than one layer, dropout_rate is also applied between the stacked layers.
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers, batch_first=True,
                           dropout=dropout_rate if n_layers > 1 else 0.0,
                           bidirectional=bidirectional)
        
        # Define dropout for regularization.
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final fully-connected layer to produce logits for each class (a bidirectional LSTM
        # outputs the forward and backward hidden states side by side).
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), n_classes)

    def train(self, mode=True):
        """
        Switch between train and eval mode; with freeze_bn the backbone's BatchNorm stays in eval.

        Args:
            mode (bool): True for training mode, False for eval mode.

        Returns:
            LRCN: self.
        """
        super().train(mode)
        if mode and self.freeze_bn:
            for module in self.base_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def freeze_backbone_stages(self, n_stages):
        """
        Freeze the stem and the first n_stages residual stages of the ResNet backbone.

        Args:
            n_stages (int): 0 = nothing frozen, 1 = stem + layer1, ..., 4 = the whole backbone.
        """
        if not 0 <= n_stages <= 4:
            raise ValueError('freeze_stages must be between 0 and 4')
        if n_stages == 0:
            return
        frozen = [self.base_model.conv1, self.base_model.bn1]
        frozen += [getattr(self.base_model, f'layer{i}') for i in range(1, n_stages + 1)]
        for module in frozen:
            for param in module.parameters():
                param.requires_grad_(False)

    def parameter_groups(self, learning_rate, backbone_lr_mult=1.0):
        """
        Optimizer parameter groups: the new layers (LSTM, classifier) and the pretrained backbone.

        The backbone already knows useful features and only needs small adjustments, so it can
        get a smaller learning rate than the randomly initialised layers. Frozen parameters are
        left out.

        Args:
            learning_rate (float): Learning rate of the LSTM and the classifier.
            backbone_lr_mult (float): Multiplier for the backbone's learning rate.

        Returns:
            list: Dicts with 'params' and 'lr'; the head first, then the backbone (if trainable).
        """
        backbone_ids = {id(p) for p in self.base_model.parameters()}
        head = [p for p in self.parameters() if id(p) not in backbone_ids and p.requires_grad]
        backbone = [p for p in self.base_model.parameters() if p.requires_grad]
        groups = [{'params': head, 'lr': learning_rate}]
        if backbone:
            groups.append({'params': backbone, 'lr': learning_rate * backbone_lr_mult})
        return groups

    def forward(self, x):
        """
        Forward pass for the LRCN model.
        
        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)
        
        The CNN backbone extracts a feature vector from every frame of every video in one call.
        The features are reshaped to (batch_size, time_steps, features) and the LSTM reads each
        video's frames as one sequence. With temporal_pooling 'last' the output at the last time
        step is passed through dropout and the final fully-connected layer to produce the class
        logits; with 'late_fusion' every time step is classified and the logits are averaged.
        Each video's logits depend only on its own frames.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        bs, ts = x.shape[:2]  # batch_size, time_steps

        # Run the CNN on all frames at once: (bs * ts, C, H, W) -> (bs * ts, features).
        frames = x.flatten(0, 1).contiguous(memory_format=torch.channels_last)
        y = self.base_model(frames)

        # One sequence of ts feature vectors per video: (bs, ts, features). One LSTM call.
        out, _ = self.rnn(self.feature_dropout(y.view(bs, ts, -1)))

        if self.temporal_pooling == 'late_fusion':
            # One prediction per frame, averaged over time (the paper's choice).
            return self.fc(self.dropout(out)).mean(dim=1)

        # Apply dropout to the output of the final time step.
        out = self.dropout(out[:, -1])
        
        # Pass the final output through the fully-connected layer to get class logits.
        out = self.fc(out)
        return out


class VideoCNN(nn.Module):
    """
    A 3D CNN for video classification, the model the hand-over's '3dcnn' type was written for.

    A 3D CNN (lecture: 3D convolution, C3D, I3D) convolves over time as well as space, so motion
    is learned inside the network instead of by an LSTM on top of per-frame features. The backbone
    is torchvision's R3D-18 pretrained on Kinetics-400, a video dataset, with its classifier
    replaced by one for our classes. The input is a clip of shape (batch, channels, time, height,
    width), which collate_fn_r3d_18 builds, at 112x112 with the Kinetics normalisation that
    transform_stats('3dcnn') returns.

    Args:
        n_classes (int): Number of output classes.
        pretrained (bool, optional): Use the Kinetics-400 weights. Default is True.
        cnn_model (str, optional): 'r3d_18' or 'r2plus1d_18'. Default is 'r3d_18'.
        freeze_bn (bool, optional): Keep the BatchNorm layers in eval mode while training (see
            LRCN). Default is False.
        freeze_stages (int, optional): Leading stages kept fixed, 1 = stem + layer1, ..., 4 = all
            (see LRCN). Default is 0.
        dropout_rate (float, optional): Dropout before the classifier. Default is 0.0.
    """
    def __init__(self, n_classes, pretrained=True, cnn_model='r3d_18', freeze_bn=False,
                 freeze_stages=0, dropout_rate=0.0):
        super().__init__()
        if cnn_model not in VIDEO_BACKBONES:
            raise ValueError('The 3D CNN backbone is not supported, choose r3d_18 or r2plus1d_18.')
        constructor, weights = VIDEO_BACKBONES[cnn_model]
        self.base_model = constructor(weights=weights if pretrained else None)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = Identity()
        self.freeze_bn = freeze_bn
        self.freeze_backbone_stages(freeze_stages)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_features, n_classes)

    def train(self, mode=True):
        """Switch modes; with freeze_bn the backbone's BatchNorm layers stay in eval mode."""
        super().train(mode)
        if mode and self.freeze_bn:
            for module in self.base_model.modules():
                if isinstance(module, nn.BatchNorm3d):
                    module.eval()
        return self

    def freeze_backbone_stages(self, n_stages):
        """
        Freeze the stem and the first n_stages residual stages of the backbone.

        Args:
            n_stages (int): 0 = nothing frozen, 1 = stem + layer1, ..., 4 = the whole backbone.
        """
        if not 0 <= n_stages <= 4:
            raise ValueError('freeze_stages must be between 0 and 4')
        if n_stages == 0:
            return
        frozen = [self.base_model.stem]
        frozen += [getattr(self.base_model, f'layer{i}') for i in range(1, n_stages + 1)]
        for module in frozen:
            for param in module.parameters():
                param.requires_grad_(False)

    def parameter_groups(self, learning_rate, backbone_lr_mult=1.0):
        """
        Optimizer parameter groups: the new classifier and the pretrained backbone (see LRCN).

        Args:
            learning_rate (float): Learning rate of the classifier.
            backbone_lr_mult (float): Multiplier for the backbone's learning rate.

        Returns:
            list: Dicts with 'params' and 'lr'; the head first, then the backbone (if trainable).
        """
        backbone = [p for p in self.base_model.parameters() if p.requires_grad]
        groups = [{'params': list(self.fc.parameters()), 'lr': learning_rate}]
        if backbone:
            groups.append({'params': backbone, 'lr': learning_rate * backbone_lr_mult})
        return groups

    def forward(self, x):
        """
        Classify a batch of clips.

        Args:
            x (Tensor): Clips of shape (batch_size, channels, time_steps, height, width).

        Returns:
            Tensor: Logits of shape (batch_size, n_classes).
        """
        return self.fc(self.dropout(self.base_model(x)))

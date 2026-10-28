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
"""

import torch
import torch.nn as nn
from torchvision import models

# ResNet constructors and their ImageNet weights. torchvision removed the ``pretrained=`` argument;
# the current API is ``weights=``.
BACKBONES = {
    'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
    'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT),
    'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
    'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT),
    'resnet152': (models.resnet152, models.ResNet152_Weights.DEFAULT),
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
    
    Raises:
        ValueError: If the specified cnn_model is not supported.
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True, cnn_model='resnet34'):
        super(LRCN, self).__init__()

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

        # Define the LSTM to process the sequence of frame features. batch_first=True makes it
        # read its input as (batch, time, features), which is what forward() builds. Without it
        # the LSTM expects (time, batch, features) and reads the batch axis as time.
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers, batch_first=True)
        
        # Define dropout for regularization.
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final fully-connected layer to produce logits for each class.
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        Forward pass for the LRCN model.
        
        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)
        
        The CNN backbone extracts a feature vector from every frame of every video in one call.
        The features are reshaped to (batch_size, time_steps, features) and the LSTM reads each
        video's frames as one sequence. The output at the last time step is passed through
        dropout and the final fully-connected layer to produce the class logits. Each video's
        logits depend only on its own frames.

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
        out, _ = self.rnn(y.view(bs, ts, -1))

        # Apply dropout to the output of the final time step.
        out = self.dropout(out[:, -1])
        
        # Pass the final output through the fully-connected layer to get class logits.
        out = self.fc(out)
        return out

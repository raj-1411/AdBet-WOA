'''
Created on Sep,2022
'''

import torch.nn as nn
import torch


class Model(nn.Module):

    def __init__(self, num_class=3):
        super(Model, self).__init__()
        self.effnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
        self.net = nn.Sequential(
                                    *list(self.effnet.children())[:-1],
                                    nn.AdaptiveAvgPool2d(output_size=1),    # (batch, 1792, 1, 1)
                                    nn.Flatten()                            # (batch, 1792)
                                )
        self.feature_count = 1792
        self.fc_layer = nn.Linear(self.feature_count, num_class)            # (batch, 3)



    def forward(self, images):
        # features generation from pre-final layer
        features = self.net(images)

        # soft-max scores
        scores = self.fc_layer(features)

        return (features, scores)
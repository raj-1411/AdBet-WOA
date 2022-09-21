'''
Created on Sep,2022
'''

import torchvision
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, num_class=2):
        super(Model, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.net = nn.Sequential(
                                    *list(self.effnet.children())[:-1],
                                    nn.AdaptiveAvgPool2d(output_size=1),    # (batch, 512, 1, 1)
                                    nn.Flatten()                            # (batch, 512)
                                )
        self.feature_count = 512
        self.fc_layer = nn.Linear(self.feature_count, num_class)            # (batch, 2)



    def forward(self, images):
        # features generation from pre-final layer
        features = self.net(images)

        # soft-max scores
        scores = self.fc_layer(features)

        return (features, scores)
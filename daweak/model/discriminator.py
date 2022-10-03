###########################################################################
# Created by: Yi-Hsuan Tsai, NEC Labs America, 2019
###########################################################################

import torch
import torch.nn as nn


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.channels = num_classes
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


def get_discriminator(num_classes=19):
    model = FCDiscriminator(num_classes)
    return model


class CWDiscriminator(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CWDiscriminator, self).__init__()

        self.models = nn.ModuleList()
        for i in range(num_classes):
            m = nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.ReLU(),
                nn.Linear(num_features, num_features),
                nn.ReLU(),
                nn.Linear(num_features, 1)
                )
            self.models.append(m)

    def forward(self, inputs):
        ''' inputs is of dimension (B, F, C) '''
        outputs = []
        for i in range(len(self.models)):
            o = self.models[i](inputs[:, :, i])
            outputs.append(o)
        outputs = torch.stack(outputs).squeeze(-1).permute(1, 0)
        return outputs


def get_classwise_discriminator(num_features, num_classes=19):
    return CWDiscriminator(num_features, num_classes)

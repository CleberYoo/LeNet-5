import torch
from torch.nn import nn
from constants import VGG16


class VGGNet(nn.Module):
    def __init__(self, model=VGG16, in_channels=3, num_classes=1000, init_weights=True):
        super(VGGNet, self).__init__()

        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(architecture=model)

        self.fcs = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(in_features=4096, out_features=4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(in_features=4096, out_features=num_classes))

        if init_weights == True:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)

        return x



    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                        
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)



    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:  # is conv layer
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(3, 3),
                                        stride=(1, 1),
                                        padding=(1, 1)),
                           nn.BatchNorm2d(num_features=x),
                           nn.ReLU()]
                
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2),
                                        stride=(2, 2))]
        
        return nn.Sequential(*layers)
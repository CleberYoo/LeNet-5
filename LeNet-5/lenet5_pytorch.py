import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Model
class LeNet5_ver1(nn.Module):
    def __init__(self, number_of_classes):
        super(LeNet5_ver1, self).__init__()         # 부모 클래스인 nn.Module의 초기화 메서드를 호출합니다. nn.Module 클래스의 모든 기능과 속성을 상속받습니다.
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
                                    nn.BatchNorm2d(num_features=6),     # original paper didn't write about BN
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
                                    nn.BatchNorm2d(num_features=16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(in_features=400, out_features=120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=84, out_features=number_of_classes)     # number_of_classes = 10

    def forward(self, input_data):
        out = self.layer1(input_data)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    
class LeNet5(nn.Module):
    """
    This model is the original one.\n
    I made exactly same with 'Gradient-Based Learning Applied to Document Recognition' Paper
    url of 'Gradient-Based Learning Applied to Document Recognition' : http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
    """
    def __init__(self, number_of_classes) -> None:
        super(LeNet5, self).__init__()

        self.ConvolutionLayer1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.SubSamplingLayer2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.ConvolutionLayer3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.SubSamplingLayer4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.ConvolutionLayer5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.FullyConnectedLayer6 = nn.Linear(in_features=120, out_features=84)
        self.FullyConnectedLayer7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, input_data):                 # input_data : 1 @ 32 X 32 X 1 image
        output_C1 = self.ConvolutionLayer1(input_data)          # 6 @ 28 X 28
        output_S2 = self.SubSamplingLayer2(output_C1)           # 6 @ 14 X 14
        output_S2 = F.sigmoid(input=output_S2)                  # 1 / (1 + exp(-1))
        output_C3 = self.ConvolutionLayer3(output_S2)           # 16 @ 10 X 10
        output_S4 = self.SubSamplingLayer4(output_C3)           # 16 @ 5 X 5
        output_C5 = self.ConvolutionLayer5(output_S4)           # 120
        output_F6 = self.FullyConnectedLayer6(output_C5)        # 84
        output = self.FullyConnectedLayer7(output_F6)           # 10

        return output

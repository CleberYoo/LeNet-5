import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F




class AlexNet(nn.Module):
    """
    ReLU non-linearity is applied to the output of every convolutional and fully-connected layer\n
    overlapped MaxPooing is applied on the 1st Conv, 2nd Conv, 5th Conv layer\n
    """
    def __init__(self):
        super().__init__()

        self.Conv1_layer = nn.Sequential(nn.Conv2d(in_channels=1,           # input : 227X227X1
                                                   out_channels=96,         # kernel : 11X11X1@96
                                                   kernel_size=(11, 11),    # output : 55X55X96
                                                   stride=4,                
                                                   padding=0                # there is no comment about padding on the paper
                                                   ),
                                         nn.ReLU(),                         # ReLU on every layer
                                         nn.MaxPool2d(kernel_size=(3, 3),   # overlapping pooling
                                                      stride=2)
                                        )
        self.Conv2_layer = nn.Sequential(nn.Conv2d(in_channels=96,          # input : 55X55X96
                                                   out_channels=256,        # kernel : 5X5@256
                                                   kernel_size=(5, 5),      # output : 27X27X256
                                                   stride=1,
                                                   padding=2),
                                         nn.ReLU(),                         # ReLU on every layer
                                         nn.MaxPool2d(kernel_size=(3, 3),   # overlapping pooling
                                                      stride=2)
                                        )
        self.Conv3_layer = nn.Sequential(nn.Conv2d(in_channels=256,
                                                   out_channels=384,
                                                   kernel_size=(3, 3),
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU()
                                                                            # NO maxpooling in the 3rd layer
                                        )
        self.Conv4_layer = nn.Sequential(nn.Conv2d(in_channels=384,
                                                   out_channels=384,
                                                   kernel_size=(3, 3),
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU()
                                                                            # NO maxpooling in the 4th layer
                                        )
        self.Conv5_layer = nn.Sequential(nn.Conv2d(in_channels=384,
                                                   out_channels=256,
                                                   kernel_size=(3, 3),
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)
                                        )
        self.FC6_layer = nn.Sequential(nn.Linear(in_features=256*6*6, out_features=4096),
                                       nn.ReLU()
                                       )
        
        self.FC7_layer = nn.Sequential(nn.Linear(in_features=4096, out_features=4096),
                                       nn.ReLU()
                                       )
        self.FC8_layer = nn.Sequential(nn.Linear(in_features=4096, out_features=10),
                                       nn.LogSoftmax(dim=1)
                                       )
    
    def forward(self, input_image):
        output_Conv1 = self.Conv1_layer(input_image)
        output_Conv2 = self.Conv2_layer(output_Conv1)
        output_Conv3 = self.Conv3_layer(output_Conv2)
        output_Conv4 = self.Conv4_layer(output_Conv3)
        output_Conv5 = self.Conv5_layer(output_Conv4)
        output_Conv5 = output_Conv5.view(output_Conv5.size(0), -1)
        
        output_FC6 = self.FC6_layer(output_Conv5)
        output_FC6 = F.dropout(input=output_FC6, p=0.5)         # Let's move these dropout to inside of the __init__

        output_FC7 = self.FC7_layer(output_FC6)
        output_FC7 = F.dropout(input=output_FC7, p=0.5)

        output_FC8 = self.FC8_layer(output_FC7)
        # output_FC8 = F.log_softmax(input=output_FC8, dim=1)
        
        return output_FC8



class AlexNet_2(nn.Module):
    """
    I put dropout in the inside of the __init__()\n

    """
    def __init__(self):
        super().__init__()

        self.Conv1_layer = nn.Sequential(nn.Conv2d(in_channels=1,           # input : 227X227X1
                                                   out_channels=96,         # kernel : 11X11X1@96
                                                   kernel_size=(11, 11),    # output : 55X55X96
                                                   stride=4,                
                                                   padding=0                # there is no comment about padding on the paper
                                                   ),
                                         nn.ReLU(),                         # ReLU on every layer
                                         nn.MaxPool2d(kernel_size=(3, 3),   # overlapping pooling
                                                      stride=2)
                                        )
        self.Conv2_layer = nn.Sequential(nn.Conv2d(in_channels=96,          # input : 55X55X96
                                                   out_channels=256,        # kernel : 5X5@256
                                                   kernel_size=(5, 5),      # output : 27X27X256
                                                   stride=1,
                                                   padding=2),
                                         nn.ReLU(),                         # ReLU on every layer
                                         nn.MaxPool2d(kernel_size=(3, 3),   # overlapping pooling
                                                      stride=2)
                                        )
        self.Conv3_layer = nn.Sequential(nn.Conv2d(in_channels=256,
                                                   out_channels=384,
                                                   kernel_size=(3, 3),
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU()
                                                                            # NO maxpooling in the 3rd layer
                                        )
        self.Conv4_layer = nn.Sequential(nn.Conv2d(in_channels=384,
                                                   out_channels=384,
                                                   kernel_size=(3, 3),
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU()
                                                                            # NO maxpooling in the 4th layer
                                        )
        self.Conv5_layer = nn.Sequential(nn.Conv2d(in_channels=384,
                                                   out_channels=256,
                                                   kernel_size=(3, 3),
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)
                                        )
        self.FC6_layer = nn.Sequential(nn.Linear(in_features=256*6*6, out_features=4096),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5)
                                       )
        
        self.FC7_layer = nn.Sequential(nn.Linear(in_features=4096, out_features=4096),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5)
                                       )
        self.FC8_layer = nn.Sequential(nn.Linear(in_features=4096, out_features=10),
                                       nn.LogSoftmax(dim=1)
                                       )
    
    def forward(self, input_image):
        output_Conv1 = self.Conv1_layer(input_image)

        output_Conv2 = self.Conv2_layer(output_Conv1)

        output_Conv3 = self.Conv3_layer(output_Conv2)

        output_Conv4 = self.Conv4_layer(output_Conv3)

        output_Conv5 = self.Conv5_layer(output_Conv4)
        output_Conv5 = output_Conv5.view(output_Conv5.size(0), -1)
        
        output_FC6 = self.FC6_layer(output_Conv5)

        output_FC7 = self.FC7_layer(output_FC6)

        output_FC8 = self.FC8_layer(output_FC7)
        
        return output_FC8
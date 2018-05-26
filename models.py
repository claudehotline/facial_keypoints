## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #(224, 224, 1) --> (222, 222, 1)
        self.conv1 = nn.Conv2d(1, 32, 3)
        #(222 ,222, 1) --> (111, 111, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        #(111, 111, 1) --> (109, 109, 1)
        self.conv2 = nn.Conv2d(32, 64, 3)
        #(109, 109, 1) --> (54, 54, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        #(54, 54, 1) --> (52, 52, 1)
        self.conv3 = nn.Conv2d(64, 128, 3)
        #(52, 52, 1) --> (26, 26, 1)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        #(26, 26, 1) --> (24, 24, 1)
        self.conv4 = nn.Conv2d(128, 256, 3)
        #(24, 24, 1) --> (12, 12, 1)
        self.maxpool4 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256*12*12, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2 * 68)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = F.dropout(self.maxpool1(F.relu(self.conv1(x))))
        x = F.dropout(self.maxpool2(F.relu(self.conv2(x))))
        x = F.dropout(self.maxpool3(F.relu(self.conv3(x))))
        x = F.dropout(self.maxpool4(F.relu(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.dropout(self.fc2(x), p=0.5)
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

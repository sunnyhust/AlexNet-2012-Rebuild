import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.create_conv_block(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, padding=0, stride=2)
        
        self.conv2 = self.create_conv_block(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, padding=0, stride=2)
        
        self.conv3 = self.create_conv_block(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = self.create_conv_block(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = self.create_conv_block(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, padding=0, stride=2)
        
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features=256*6*6, out_features=4096), nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features=4096, out_features=4096), nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=60)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool5(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    def create_conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding= padding, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()

        )
    


import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,64,7),
            nn.ReLU(True),
            nn.Conv2d(64,64,7),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,32,7),
            nn.ReLU(True),
            nn.Conv2d(32,32,7),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,16,7),
            nn.ReLU(True),
            nn.Conv2d(16,16,7),
            nn.ReLU(True),
        )

        self.linear = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(16 * 43 * 43,10000),
            nn.ReLU(True),
            nn.Linear(10000,2000),
            nn.ReLU(True),
            nn.Linear(2000,500),
            nn.ReLU(True),
            nn.Linear(500,230)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,16 * 43 * 43)
        x = self.linear(x)
        
        return x
    

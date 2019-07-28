import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,64,5),
            nn.ReLU(True),
            nn.Conv2d(64,64,5),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),
            nn.Conv2d(64,32,5),
            nn.ReLU(True),
            nn.Conv2d(32,32,5),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),
            nn.Conv2d(32,16,5),
            nn.ReLU(True),
            nn.Conv2d(16,16,5),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),
            nn.Conv2d(16,8,5),
            nn.ReLU(True),
            nn.Conv2d(8,8,5),
            nn.ReLU(True),
        )

        self.linear = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(8 * 49 * 49,3000),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(3000,500),
            nn.ReLU(True),
            nn.Dropout(0.25),
            #nn.Linear(2000,500),
            #nn.ReLU(True),
            #nn.Dropout(0.25),
            nn.Linear(500,230)
        )

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,8*49*49)
        x = self.linear(x)
        #x = self.softmax(x)

        return x

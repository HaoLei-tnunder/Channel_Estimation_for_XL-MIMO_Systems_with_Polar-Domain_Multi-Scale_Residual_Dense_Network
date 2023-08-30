from models.subNets import *
from models.cbam import *


class MRDN(nn.Module):
    def __init__(self, input_channel, numofmodules=2, numforrg=4, numofrdb=10, numofconv=6, numoffilters=60, t=1):
        super(MRDN, self).__init__()

        self.numofmodules = numofmodules # num of modules to make residual
        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units
        self.nDenselayer = numofconv
        self.numofkernels = numoffilters
        self.t = t
        
        # self.layer0 = nn.BatchNorm2d(num_features = 80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1 = nn.Conv2d(input_channel, self.numofkernels, kernel_size=3, stride=1, padding=1)
        # self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1)

        modules = []
        for i in range(self.numofrdb // (self.numofmodules * self.numforrg)  ):   # 10 // ( 2 * 4) = 1
            modules.append(GGRDB(self.numofmodules, self.numofkernels, self.nDenselayer, self.numofkernels, self.numforrg)) # 2,filter,6,filter,4
       
        self.rglayer = nn.Sequential(*modules)
        self.layer7 = nn.ConvTranspose2d(self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1)
        # self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(self.numofkernels, input_channel, kernel_size=3, stride=1, padding=1)
        self.cbam = CBAM(numoffilters, 16)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer3(out)

        for grdb in self.rglayer:
            for i in range(self.t):
                out = grdb(out)

        out = self.layer7(out)   
        out = self.cbam(out)
        out = self.layer9(out)

        return out


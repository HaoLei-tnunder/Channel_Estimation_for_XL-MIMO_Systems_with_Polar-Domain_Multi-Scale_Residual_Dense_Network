import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ASPP import ASPP



####################################################################################################################

class make_dense(nn.Module):
    def __init__(self, nChannels, nChannels_, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels_, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
        # self.relu = nn.LeakyReLU(inplace=True)
        self.nChannels = nChannels

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, nChannels, nDenselayer, growthRate):    # filter,6,filter
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    # 6
            modules.append(make_dense(nChannels, nChannels_, growthRate))
            # modules.append(make_dense_LReLU(nChannels, growthRate))
            nChannels_ += growthRate
        # self.aspp = ASPP( 80 )
        self.dense_layers = nn.Sequential(*modules)

        ###################kingrdb ver2##############################################
        # self.conv_1x1 = nn.Conv2d(nChannels_ + growthRate, nChannels, kernel_size=1, padding=0, bias=False)
        ###################else######################################################
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)


    def forward(self, x):
        out = self.dense_layers(x)

        # local residual 구조
        out = self.conv_1x1( out )
        out = out + x
        return out

class RDB_ms(nn.Module):   #multi-scale
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, nChannels, nDenselayer, growthRate):
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(RDB_ms, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels, nChannels_, growthRate))
            # modules.append(make_dense_LReLU(nChannels, growthRate))
            nChannels_ += growthRate
        self.aspp = ASPP( growthRate )
        self.dense_layers = nn.Sequential(*modules)
       
        self.conv_1x1 = nn.Conv2d(nChannels_ + growthRate , nChannels, kernel_size=1, padding=0, bias=False)


    def forward(self, x):
        out = self.dense_layers(x)
        outaspp = self.aspp(x)
        outputlist=[]
        outputlist.append(out)
        outputlist.append(outaspp)
        concat = torch.cat(outputlist, 1)
            
        # local residual 구조
        out = self.conv_1x1( concat )
        out = out + x
        return out



####################################################################################################################
# Group of Residual dense block (GRDB) architecture
class GRDB(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, numofkernels, nDenselayer, growthRate, numforrg):   # filter,6,filter,4
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(GRDB, self).__init__()

        numforrg = numforrg 
        nDenselayer = nDenselayer 

        modules = []
        for i in range(numforrg   ):   # 4
            # modules.append(RDB(numofkernels, nDenselayer=nDenselayer, growthRate=growthRate))   # filter,6,filter
            modules.append(RDB_ms(numofkernels, nDenselayer=nDenselayer, growthRate=growthRate))
        self.rdbs = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(numofkernels * numforrg, numofkernels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        out = x
        outputlist = []
        for rdb in self.rdbs:
            output = rdb(out)
            outputlist.append(output)
            out = output
        concat = torch.cat(outputlist, 1)
        out = x + self.conv_1x1(concat)
        return out


# Group of group of Residual dense block (GRDB) architecture
class GGRDB(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, numofmodules, numofkernels, nDenselayer, growthRate, numforrg):   # 2,filter,6,filter,4
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(GGRDB, self).__init__()

        modules = []
        for i in range(numofmodules   ):   # 2
            modules.append(GRDB(numofkernels, nDenselayer=nDenselayer, growthRate=growthRate, numforrg=numforrg)) # filter,6,filter,4
        self.grdbs = nn.Sequential(*modules)

    def forward(self, x):
        output = x
        for grdb in self.grdbs:
            output = grdb(output)

        return x + output


####################################################################################################################


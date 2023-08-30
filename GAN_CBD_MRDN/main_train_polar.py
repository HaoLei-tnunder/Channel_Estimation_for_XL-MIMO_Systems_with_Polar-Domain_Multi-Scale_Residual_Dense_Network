import h5py
import scipy.io as sio
import numpy as np
import torch.nn as nn
import torch
# import torchvision
import torch.utils.data as data
from torch.autograd import Variable
import math
import platform
import time
from scipy.fftpack import fft2, ifft2, fft, ifft
from models.DenoisingModels import MRDN
# from GAN_model import G_CBDNet
# from Res_CD_Net import D_discriminater_Net
from functional import compute_SNR, fft_reshape, fft_shrink, add_noise_improve, real_imag_stack_polar, Noise_map, \
    tensor_reshape, ifft_tensor_polar, compute_NMSE, test, nomal_noiselevel, polar_reshape, compute_NMSE_linear
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDE_VISIBLE_DIVICES"] = "1"

EPOCH = 100
BATCH_SIZE = 80
LR = 0.00001
focus = 0.1
img_height = 128
img_width = 1
img_channels = 1
Max_abs = 100

Step = 0

mat = h5py.File('../data/train.mat','r')         #读取文件，得到字典
x_train = mat['H']                      #获取H_ori数据
x_train = np.transpose(x_train)
print(np.shape(x_train)) 

mat1 = h5py.File('../data/Polarcodebook.mat','r')         
Polarcodebook = mat1['Polarcodebook']                  
Polarcodebook = np.transpose(Polarcodebook)
print(np.shape(Polarcodebook)) 

Polarcodebook = polar_reshape(Polarcodebook)
SS = Polarcodebook.shape[1]
uu,ss,vv = np.linalg.svd(Polarcodebook)
ss =1./ss
ss = np.diag(  ss  )
S = np.zeros([SS, 128], dtype=float) 
S[0:128,0:128] =  ss
x1 =  np.dot(  np.dot( np.transpose(vv) , S )    , np.transpose(uu) ).conjugate()

NNNMSE = np.zeros([ 1 ,EPOCH  ], dtype=float) 

x, y, H, H_get = fft_reshape(x_train, img_height, img_width)
#print(np.shape(H_get))       # 4000 64 32


train_loader = data.DataLoader(dataset=H_get, batch_size=BATCH_SIZE, shuffle=True)

G_net = MRDN(input_channel=1, numoffilters=80, t=1)
device_ids = [0]
# print(G_net)
G = nn.DataParallel(G_net, device_ids=device_ids).cuda()

Loss = nn.MSELoss()
Loss.cuda()

criterion = nn.BCELoss()  # 是单目标二分类交叉熵函数
g_optimizer = torch.optim.Adam(G.parameters(), lr=LR)


G.load_state_dict(torch.load('../model/P_MSRDN_0.0001_10_30_200_5m_50m.pt'))


for epoch in range(EPOCH):


    print('epoch = ', epoch)

    tr_loss = 0.0
    running_loss = 0.0

    for i, x in enumerate(train_loader, 0):

        # T1 = time.perf_counter()

        NN = len(x)                         #BATCH_SIZE
     
        real_label = Variable(torch.ones(NN)).cuda()  # 定义真实的图片label为1
        fake_label = Variable(torch.zeros(NN)).cuda()  # 定义假的图片的label为0

        sx = x.numpy()                                                            

        H_train_data = np.zeros([NN, img_height, img_width * 2], dtype=complex)    
        H_train_data = H_train_data + sx 

        H = fft_shrink(H_train_data, img_height, img_width)                       

        noise, E_output1 = add_noise_improve(H, 10, 30)                          

        H_n = noise + H
        SNR = compute_SNR(  H, noise )

        # polar
        H_n_fft_p = np.zeros([BATCH_SIZE, SS, 1], dtype=complex)                  # 64 1
        H_fft_p = np.zeros([BATCH_SIZE, SS, 1], dtype=complex)

        # T1 = time.perf_counter()
        for i_num in range(BATCH_SIZE):
            H_n_fft_p[i_num, :, :] = np.dot( x1, H_n[i_num, :, :] )       
            H_fft_p[i_num, :, :] = np.dot( x1, H[i_num, :, :] )         


        # 64x32->64x64 numpy real+imag
        H_n_fft_r_i = real_imag_stack_polar(H_n_fft_p)                                  #64  64
        H_n_fft_stack = tensor_reshape(H_n_fft_r_i)


        noise_r_i = H_n_fft_r_i - real_imag_stack_polar(H_fft_p)
        noise_stack = tensor_reshape(noise_r_i)

        H_n_fft_train = Variable(H_n_fft_stack.cuda())     
        real_img = Variable(noise_stack.cuda())  # H

        fake_img = G(H_n_fft_train)  # 随机噪声输入到生成器中，得到一副假的图片  

        g_loss = Loss(fake_img, real_img)*10000

        # bp and optimize
        g_optimizer.zero_grad()  # 梯度归0
        g_loss.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

        h_fft_pre = torch.zeros([BATCH_SIZE, SS, 2])

        for i_num in range(BATCH_SIZE):
            h_fft_pre[i_num, :, :] = H_n_fft_train[i_num, :, :] - fake_img[i_num, :, :] 


        ssx = h_fft_pre.detach().numpy()
        ssx_i = np.zeros([NN, SS, 2], dtype=complex)  # 64 X 64
        ssx_i = ssx + ssx_i 
        H_fft_pre_last = fft_shrink(ssx_i, SS, img_width)     
        H_fft_pre_last1 = ifft_tensor_polar(H_fft_pre_last , Polarcodebook)

        NMSE = compute_NMSE( H_fft_pre_last1, H)   

        print('NMSE = ',NMSE, ',')

        NNNMSE[:, epoch] = NMSE / 250 + NNNMSE[:, epoch] 

              
        if i % 20 == 19:
            tr_loss = running_loss / 20

        if Step % 20 == 0:
            # G.apply(svd_orthogonalization)
            print("[epoch - polar %d][%d/%d] g_loss: %.4f SNR: %.4f NMSE: %.4f" % (epoch + 1, i + 1, len(train_loader), g_loss,
                                                                          SNR, NMSE))
            # print('NMSE = ',NMSE, ',')
        Step += 1

torch.save(G.state_dict(), '../model/P_MSRDN.pt')



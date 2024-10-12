from copy import deepcopy
import random
import math
from tokenize import Double
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from models.build_gen import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
from matplotlib import pyplot as plt
from dataset.dataset_read import dataset_read
from dataset.dataset_read36 import *
from models.loss import *
from utils.utils import *
from utils.sub import *
from torchstat import stat
import datetime

seed_all(3407)
decon = nn.Linear(70, 1)
 # 定义一个保存函数，添加到 CSV 文件中
def save_to_csv(data, filepath):
    """
    将损失数据保存到 CSV 文件中，追加模式。
    每次保存一行数据。
    """
    with open(filepath, 'a') as f:
        np.savetxt(f, [data], delimiter=',')  # 使用 [] 包裹 data，确保每次写入的是一行
def preprocess_data(data, label):
    data = data.cuda().to(torch.float32)
    label = label.cuda().to(torch.float32)
    return data, (label).cpu().detach().numpy()

def get_output(data, index):
    u, u_mu, u_log_var = UNet(data, test_seq[index])
    u_inv, u_inv_var, up_inv, up_inv_var = Uinv(u)
    u_spf,_,_,_ = Uspf(u)
    # q_z, q_z_mu, q_z_log_var= Q_ZNet(data, u_inv)
    q_z, q_z_mu, q_z_log_var, p_z, p_z_mu, p_z_log_var= Q_ZNet(data, u_inv)
    output = PredNet(q_z)
    return (output).cpu().detach().numpy(),u_inv,u_spf,u

def calculate_metrics(output, label):
    MAE = mean_absolute_error(output, label)
    RMSE = sqrt(mean_squared_error(output, label))
    R2 = r2_score(output, label)
    return MAE, RMSE, R2
    
device = torch.device("cuda:0")
batch_size = 64    #64
lr = 5e-4    #5e-4
n_epoch = 1601 #851,1401
interval = 50 
num = 1
weight_decay = 0
lambda_pre = 1
num_domain = 4
d_loss_type = "DANN_loss_mean"
lambda_gan = 0.5  #0.5
lambda_u_concentrate = 0.5  #0.5
lamada_z = 1e-5    #1e-5
lambda_z_concentrate = 0.5   # 0.5
lambda_spf = 1#1
beta = 1

savemodel = False
loadmodel = False

df1 = pd.read_csv('/root/dh/2023/so2/data/tarla4.csv')
df2 = pd.read_csv('/root/dh/2023/so2/data/tarla8.csv')
df3 = pd.read_csv('/root/dh/2023/so2/data/tarla12.csv')
df4 = pd.read_csv('/root/dh/2023/so2/data/tarla16.csv')
label = pd.read_csv('/root/dh/2023/so2/data/target.csv')
datasets,x_val, y_val = dataset_read36(
    df1,df2,df3,df4, label = label, batch_size = batch_size)

t = to_tensor(np.linspace(0, 1, 4).astype(np.float32))
z = 1/4

x_seq = []
d_seq = []
t_seq = to_tensor(np.zeros((4, batch_size, 1), dtype=np.float32)) + t.reshape(4, 1, 1)
test_seq = to_tensor(np.zeros((4, 120, 1), dtype=np.float32)) + t.reshape(4, 1, 1)
z_seq = to_tensor(np.zeros((4, batch_size, 1), dtype=np.float32)) + z
cuda = True
cudnn.benchmark = True

model_path = ''
decon = decon.to(device)
UNet = VDI('UNet').to(device)
Uinv = VDI('uinv').to(device)
Uspf = VDI('uspf').to(device)
Q_ZNet = VDI('Q_ZNetban1').to(device)
PredNet = VDI('PredNet').to(device)
ReconstructNet = VDI('ReconstructNet').to(device)
Z_ReconstructNet = VDI('ZReconstructNet').to(device)
U_ReconstructNet = VDI('UReconstructNet').to(device)
SAR = VDI('sar').to(device)
netD = VDI('ClassDiscNet').to(device)

if loadmodel:
    UNet.load_state_dict(torch.load(model_path+'/UNet.pth'))
    Uinv.load_state_dict(torch.load(model_path+'/Uinv.pth'))
    Q_ZNet.load_state_dict(torch.load(model_path+'/Q_ZNet.pth'))
    PredNet.load_state_dict(torch.load(model_path+'/PredNet.pth'))

loss_predict = torch.nn.MSELoss(reduction='mean')
crossentropyloss=nn.CrossEntropyLoss()
UZF_parameters = list(UNet.parameters()) + list(
                Q_ZNet.parameters()) + list(PredNet.parameters()) + list(
                    ReconstructNet.parameters()) + list(
                    Uinv.parameters()) + list(Z_ReconstructNet.parameters()) + list(U_ReconstructNet.parameters()) + list(
                    Uspf.parameters())
U_parameters = list(UNet.parameters()) + list(
                Q_ZNet.parameters()) + list(
                    ReconstructNet.parameters()) + list(
                    Uinv.parameters()) + list(Z_ReconstructNet.parameters()) + list(
                    Uspf.parameters())
optimizer_U = optim.Adam(U_parameters, lr=lr, weight_decay=weight_decay)
optimizer_UZF = optim.Adam(UZF_parameters, lr=lr, weight_decay=weight_decay)
optimizer_D = optim.Adam(netD.parameters(), lr=lr, weight_decay=weight_decay)
# note u == \alpha
for epoch in range(n_epoch):
    for batch_idx, ((x_seq0, y_seq0), (x_seq1, y_seq1),(x_seq2, y_seq2),(x_seq3, y_seq3)) in enumerate(zip(datasets[0], datasets[1], datasets[2], datasets[3])):
        x_seq = torch.stack([x_seq0.cuda(), x_seq1.cuda(), x_seq2.cuda(), x_seq3.cuda()], dim=0)
        y_lable = torch.stack([y_seq0.cuda(), y_seq1.cuda(), y_seq2.cuda(), y_seq3.cuda()], dim=0)
        x_seq = x_seq.to(torch.float32)
        y_lable = y_lable.to(torch.float32)
        
        set_requires_grad(netD, requires_grad=True)
        optimizer_U.zero_grad()
        optimizer_D.zero_grad()
        u, u_mu, u_log_var = UNet(x_seq, t_seq)   # u, u_mu, u_log_var 6*128*2
        u_inv, u_inv_var, up_inv, up_inv_var = Uinv(u)
        q_z, q_z_mu, q_z_log_var, p_z, p_z_mu, p_z_log_var = Q_ZNet(x_seq, u_inv)    # 6*128*204, 6*128*2, 6*128*1
        d = netD(q_z)
        loss_D = F.l1_loss(flat(d), flat(t_seq))
        loss_D.backward()
        optimizer_D.step()

        set_requires_grad(netD, requires_grad=False)
        optimizer_UZF.zero_grad()
        u, u_mu, u_log_var = UNet(x_seq, t_seq)   # u, u_mu, u_log_var 6*128*2
        u_inv, u_inv_var, up_inv, up_inv_var = Uinv(u)
        u_spf, _, _, _ = Uspf(u)
        q_z, q_z_mu, q_z_log_var, p_z, p_z_mu, p_z_log_var = Q_ZNet(x_seq, u_inv) 
        u_x = U_ReconstructNet(u_inv)
        r_x = ReconstructNet(u)
        z_x = Z_ReconstructNet(q_z)
        y_seq = PredNet(q_z)
        d = netD(q_z)
        d_inv = SAR(u_inv)
        d_spf = SAR(u_spf)
        loss_spf = crossentropyloss(d_spf, t_seq)
        # loss_spf = F.l1_loss(flat(d_spf), flat(t_seq))
        loss_inv1 = F.kl_div(d_inv.softmax(dim=1).log(), z_seq.softmax(dim=1), reduction='sum')
        loss_inv2 = F.kl_div(z_seq.softmax(dim=1).log(), d_inv.softmax(dim=1), reduction='sum')
        # loss_inv = crossentropyloss(d_inv, z_seq)
        loss_inv = (loss_inv1+loss_inv2)/2
        # E_q[log p(y|z)]
        loss_p_y_z = loss_predict(flat(y_seq), flat(y_lable))
        # reconstruction loss (p(x|u))
        loss_p_x_u = loss_predict(flat(x_seq), flat(r_x))
        # reconstruction loss (p(x|z))
        loss_p_x_z = loss_predict(flat(x_seq), flat(z_x))
        loss_p_u_uinv = loss_predict(flat(u_x), flat(u_inv))
        # gan loss (adversarial loss)
        loss_E_gan = - F.l1_loss(flat(d), flat(t_seq))
        # KL loss
        loss_KLu = -0.5 * torch.sum(1 + u_log_var - u_mu.pow(2) - u_log_var.exp())
        # loss_KLz = -0.5 * torch.sum(1 + q_z_log_var - q_z_mu.pow(2) - q_z_log_var.exp())
        loss_p_z_x_u = -0.5 * flat(p_z_log_var) - 0.5 * (
            torch.exp(flat(q_z_log_var)) +
            (flat(q_z_mu) - flat(p_z_mu))**2) / flat(
                torch.exp(p_z_log_var))
        loss_KLz = -torch.mean(loss_p_z_x_u.sum(1), dim=0)
        
        loss_uinv = -0.5 * flat(up_inv_var) - 0.5 * (
            torch.exp(flat(u_inv_var)) +
            (flat(u_inv) - flat(up_inv))**2) / flat(
                torch.exp(up_inv_var))
        loss_KLuinv = -torch.mean(loss_uinv.sum(1), dim=0)

        # - E_q[log q(u|x)] 熵
        # u is multi-dimensional
        loss_q_u_x = torch.mean((0.5 * flat(u_log_var)).sum(1), dim=0)

        # - E_q[log q(z|x,u)]
        # remove all the losses about log var and use 1 as var
        loss_q_z_x_u = torch.mean((0.5 * flat(q_z_log_var)).sum(1), dim=0)

        loss_E = loss_E_gan * lambda_gan + lambda_pre*loss_p_y_z + lambda_z_concentrate*loss_p_x_z + lambda_u_concentrate*loss_p_x_u + (loss_KLu + 0.1*loss_KLuinv +
                lamada_z*loss_KLz + lambda_spf*loss_spf + loss_inv2)
        loss_E.backward()

        optimizer_UZF.step()
        
    if epoch % interval == 0 and epoch != 0:   # 第interval轮全部训练完
        
        print('Train Epoch: {}\t  Loss_D: {:.6f}\t  loss_E: {:.6f}\t   loss_spf: {:.6f}\t  loss_inv: {:.6f}\t'.format(
            epoch,  loss_D.data, loss_E.data, loss_spf.data, loss_inv.data))  # 打印源域预测损失和总的损失
        
        for i in range(4):
            data, label = preprocess_data(x_val[i], y_val[i])
            output,u_inv,u_spf,_ = get_output(data, i)


data, label = preprocess_data(x_val[0], y_val[0])
output, u_inv,u_spf,_ = get_output(data, 0)

MAE, RMSE, R2 = calculate_metrics(output, label)
print('MAE: {} \t RMSE: {} \t '.format(MAE, RMSE))


# current_time = datetime.datetime.now()
# print(f"VDISDA聚酯,当前时间：{current_time}, Batch size: {batch_size}, 学习率: {lr}, epoch: {n_epoch}, gan: {lambda_gan}, lamada_z: {lamada_z}, lambda_z_concentrate:{lambda_z_concentrate}, lambda_u_concentrate:{lambda_u_concentrate},lambda_spf: {lambda_spf}")


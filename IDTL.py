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
from source_only import Trainer
from loss import *
from models.loss import *
from utils.utils import *
from utils.sub import *
from torchstat import stat
import datetime
import time

seed_all(3407)
decon = nn.Linear(70, 1)
def save_to_csv(data, filepath):
    with open(filepath, 'a') as f:
        np.savetxt(f, [data], delimiter=',') 
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

def errplt(predictions, labels):
    model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
    num_models = len(predictions)
    colors = ['b', 'g', 'r', 'c', 'm', 'y'] 
    
    plt.figure(dpi=300,figsize=(10, 6))
    for i in range(num_models):
        errors = predictions[i] - labels
        plt.plot(errors, marker='o', linestyle='-', color=colors[i], label=model_names[i])
        
    plt.axhline(y=0, color='k', linestyle='--')  
    # plt.ylim(-0.6, 0.4)
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Errors between True Values and Predicted Values')
    plt.legend()
    plt.savefig(f"/root/dh/2023/so2/figs/err.png")
    plt.close()

def zone_and_link(x_val, y_val):
    inv = []
    spf = []
    for i in range(4):
        data, label = preprocess_data(x_val[i], y_val[i])
        output, u_inv,u_spf,u = get_output(data, i)
        con = decon(u_inv).squeeze(-1)
        inv.append(con.cpu().detach().numpy())
        conspf = decon(u_spf).squeeze(-1)
        spf.append(conspf.cpu().detach().numpy())
    inv0, inv1,inv2,inv3= inv[0],  inv[1], inv[2], inv[3], 
    r = len(inv0) + 1

    fig, ax = plt.subplots()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font', family='Times New Roman')
    plt.figure(dpi=300, figsize=(14, 8))

    plt.ylim(0.01, 0.04)

    plt.plot(np.arange(1, r), inv0, 'r-', label=r"$\beta_{din}$ 1")
    plt.plot(np.arange(1, r), inv1, 'g-', label=r"$\beta_{din}$ 2")
    plt.plot(np.arange(1, r), inv2, 'b-', label=r"$\beta_{din}$ 3")
    plt.plot(np.arange(1, r), inv3, 'm-', label=r"$\beta_{din}$ 4")
    plt.xlabel('Sample Point', size=28)
    plt.ylabel('Output', size=28)
    plt.tick_params(labelsize=24)
    plt.legend(fontsize=22)

    axins = ax.inset_axes((0.45, 0.1, 0.2, 0.2))

    axins.plot(np.arange(1, r), inv0, color='#000000', linewidth='1', marker='>',
                label=r"$\beta_{inv}$ 1")
    axins.plot(np.arange(1, r), inv1, color='#4b6cff', linewidth='1', marker='s',
                label=r"$\beta_{inv}$ 2")
    axins.plot(np.arange(1, r), inv2, color='#FFBD23', linewidth='1', marker='p',
                label=r"$\beta_{inv}$ 3")
    axins.plot(np.arange(1, r), inv3, color='r', linewidth='1', marker='*',
                label=r"$\beta_{inv}$ 4")

    zone_left = 40
    zone_right = 60

    axins.set_xlim(40, 60)
    axins.set_ylim(0.01, 0.04)

    zone_and_linked(ax, axins, zone_left, zone_right, np.arange(1, r), [inv0, inv1, inv2, inv3], 'right')

    plt.legend(fontsize=22)
    plt.show()  
    plt.savefig(f"/root/dh/2023/so2/inv1.png")
    plt.savefig(f"/root/dh/2023/so2/inv1.eps")
    
def plot_inv_spf(x_val, y_val):
    inv = []
    spf = []
    for i in range(4):
        data, label = preprocess_data(x_val[i], y_val[i])
        output, u_inv,u_spf,u = get_output(data, i)
        con = decon(u_inv).squeeze(-1)
        inv.append(con.cpu().detach().numpy())
        conspf = decon(u_spf).squeeze(-1)
        spf.append(conspf.cpu().detach().numpy())
    
    r = len(inv[0]) + 1
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font',family='Times New Roman')
    plt.figure(dpi=300, figsize=(14, 8))
    
    plt.ylim(0.01, 0.04)
    
    plt.plot(np.arange(1, r), inv[0], 'r-', label=r"$\beta_{inv}$ 1")
    plt.plot(np.arange(1, r), inv[1], 'g-', label=r"$\beta_{inv}$ 2")
    plt.plot(np.arange(1, r), inv[2], 'b-', label=r"$\beta_{inv}$ 3")
    plt.plot(np.arange(1, r), inv[3], 'm-', label=r"$\beta_{inv}$ 4")
    plt.xlabel('Sample Point', size=28)
    plt.ylabel('Output', size=28)
    plt.tick_params(labelsize=24)
    plt.legend(fontsize=22)
    plt.savefig(f"/root/dh/2023/so2/figs/inv1.png")
    plt.savefig(f"/root/dh/2023/so2/figs/inv1.eps")
    plt.close()

    r = len(spf[0]) + 1
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font',family='Times New Roman')
    plt.figure(dpi=300, figsize=(14, 8))
    # plt.ylim(0.013, 0.016)
    plt.plot(np.arange(1, r), spf[0], 'r-', label=r"$\beta_{dsp}$ 1")
    plt.plot(np.arange(1, r), spf[1], 'g-', label=r"$\beta_{dsp}$ 2")
    plt.plot(np.arange(1, r), spf[2], 'b-', label=r"$\beta_{dsp}$ 3")
    plt.plot(np.arange(1, r), spf[3], 'm-', label=r"$\beta_{dsp}$ 4")
    plt.xlabel('Sample Point', size=28)
    plt.ylabel('Output', size=28)
    plt.tick_params(labelsize=24)
    plt.legend(fontsize=22)
    plt.savefig(f"/root/dh/2023/so2/figs/spf.png")
    plt.savefig(f"/root/dh/2023/so2/figs/spf.eps")
    plt.close()
    
device = torch.device("cuda:0")
batch_size = 64    #64
lr = 5e-4    #5e-4
n_epoch = 1401 #851,1401
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

model_path = '/root/dh/2023/so2/model_path'
decon = decon.to(device)
UNet = IDTL('UNet').to(device)
Uinv = IDTL('uinv').to(device)
Uspf = IDTL('uspf').to(device)
Q_ZNet = IDTL('Q_ZNetban1').to(device)
PredNet = IDTL('PredNet').to(device)
ReconstructNet = IDTL('ReconstructNet').to(device)
Z_ReconstructNet = IDTL('ZReconstructNet').to(device)
U_ReconstructNet = IDTL('UReconstructNet').to(device)
SAR = IDTL('sar').to(device)
netD = IDTL('ClassDiscNet').to(device)
time_start = time.time()
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

for epoch in range(n_epoch):
    for batch_idx, ((x_seq0, y_seq0), (x_seq1, y_seq1),(x_seq2, y_seq2),(x_seq3, y_seq3)) in enumerate(zip(datasets[0], datasets[1], datasets[2], datasets[3])):
        x_seq = torch.stack([x_seq0.cuda(), x_seq1.cuda(), x_seq2.cuda(), x_seq3.cuda()], dim=0)
        y_lable = torch.stack([y_seq0.cuda(), y_seq1.cuda(), y_seq2.cuda(), y_seq3.cuda()], dim=0)
        x_seq = x_seq.to(torch.float32)
        y_lable = y_lable.to(torch.float32)
        
        set_requires_grad(netD, requires_grad=True)
        optimizer_U.zero_grad()
        optimizer_D.zero_grad()
        u, u_mu, u_log_var = UNet(x_seq, t_seq)   
        u_inv, u_inv_var, up_inv, up_inv_var = Uinv(u)
        q_z, q_z_mu, q_z_log_var, p_z, p_z_mu, p_z_log_var = Q_ZNet(x_seq, u_inv)    
        d = netD(q_z)
        loss_D = F.l1_loss(flat(d), flat(t_seq))
        loss_D.backward()
        optimizer_D.step()

        set_requires_grad(netD, requires_grad=False)
        optimizer_UZF.zero_grad()
        u, u_mu, u_log_var = UNet(x_seq, t_seq)  
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
        loss_inv1 = F.kl_div(d_inv.softmax(dim=1).log(), z_seq.softmax(dim=1), reduction='sum')
        loss_inv2 = F.kl_div(z_seq.softmax(dim=1).log(), d_inv.softmax(dim=1), reduction='sum')
        loss_inv = (loss_inv1+loss_inv2)/2
        loss_p_y_z = loss_predict(flat(y_seq), flat(y_lable))
        loss_p_x_u = loss_predict(flat(x_seq), flat(r_x))
        loss_p_x_z = loss_predict(flat(x_seq), flat(z_x))
        loss_p_u_uinv = loss_predict(flat(u_x), flat(u_inv))
        loss_E_gan = - F.l1_loss(flat(d), flat(t_seq))
        loss_KLu = -0.5 * torch.sum(1 + u_log_var - u_mu.pow(2) - u_log_var.exp())
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
        loss_q_u_x = torch.mean((0.5 * flat(u_log_var)).sum(1), dim=0)
        loss_q_z_x_u = torch.mean((0.5 * flat(q_z_log_var)).sum(1), dim=0)
        loss_E = loss_E_gan * lambda_gan + lambda_pre*loss_p_y_z + lambda_z_concentrate*loss_p_x_z + lambda_u_concentrate*loss_p_x_u + (loss_KLu + 0.1*loss_KLuinv +
                lamada_z*loss_KLz + lambda_spf*loss_spf + loss_inv2)
        loss_E.backward()

        optimizer_UZF.step()
        
    if epoch % interval == 0 and epoch != 0: 
        
        print('Train Epoch: {}\t  Loss_D: {:.6f}\t  loss_E: {:.6f}\t   loss_spf: {:.6f}\t  loss_inv: {:.6f}\t'.format(
            epoch,  loss_D.data, loss_E.data, loss_spf.data, loss_inv.data))
        
        outputs = []
        labels = []
        for i in range(4):
            data, label = preprocess_data(x_val[i], y_val[i])
            output,u_inv,u_spf,_ = get_output(data, i)
            outputs.append(output)
            labels.append(label)

        for i in range(4):
            MAE, RMSE, R2 = calculate_metrics(outputs[i], labels[i])
            print('MAE C{}: {} \t RMSE C{}: {} \t  R2 C{}: {} \t '.format(i, MAE, i, RMSE, i, R2))

        
outputs = []
labels = []
for i in range(4):
    data, label = preprocess_data(x_val[i], y_val[i])
    output, u_inv,u_spf,_ = get_output(data, i)
    outputs.append(output)
    labels.append(label)
time_end = time.time()
for i in range(4):
    MAE, RMSE, R2 = calculate_metrics(outputs[i], labels[i])
    print('MAE C{}: {} \t RMSE C{}: {} \t  R2 C{}: {} \t '.format(i, MAE, i, RMSE, i, R2))

for i in range(len(outputs)):
    r = outputs[i].shape[0] + 1
    plt.plot(np.arange(1, r), labels[i], 'r-', label="real")
    plt.plot(np.arange(1, r), outputs[i], 'g-', label="pred")
    plt.legend()
    plt.savefig(f"/root/dh/2023/so2/figs/mul{i}.png")
    plt.close()

time_sum = time_end - time_start
print(time_sum)
current_time = datetime.datetime.now()
print(f"IDTL on debutanizer column,当前时间：{current_time}, Batch size: {batch_size}, 学习率: {lr}, epoch: {n_epoch}, gan: {lambda_gan}, lamada_z: {lamada_z}, lambda_z_concentrate:{lambda_z_concentrate}, lambda_u_concentrate:{lambda_u_concentrate},lambda_spf: {lambda_spf}")

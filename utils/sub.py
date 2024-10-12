import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
from matplotlib.patches import  ConnectionPatch
import numpy as np

def zone_and_linked(ax,axins,zone_left,zone_right,x,y,linked='bottom',
                    x_ratio=0.05,y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data)-(np.max(y_data)-np.min(y_data))*y_ratio
    ylim_top = np.max(y_data)+(np.max(y_data)-np.min(y_data))*y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
            [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_left,ylim_bottom)
        xyA_2, xyB_2 = (xlim_right,ylim_top), (xlim_right,ylim_bottom)
    elif  linked == 'top':
        xyA_1, xyB_1 = (xlim_left,ylim_bottom), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_right,ylim_top)
    elif  linked == 'left':
        xyA_1, xyB_1 = (xlim_right,ylim_top), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_left,ylim_bottom)
    elif  linked == 'right':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_right,ylim_top)
        xyA_2, xyB_2 = (xlim_left,ylim_bottom), (xlim_right,ylim_bottom)
        
    con = ConnectionPatch(xyA=xyA_1,xyB=xyB_1,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2,xyB=xyB_2,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)

# ################预测曲线
# data1 = pd.read_csv('PRED.csv').values
# w1 = 500
# SAE = data1[:w1, 0]
# SEAE = data1[:w1, 1]
# SGTAE = data1[:w1, 2]
# LIDATE = data1[:w1, 3]
# SISAE_GRU = data1[:w1, 4]
# MDTSGAE = data1[:w1, 3]
# REAL = data1[:w1, 6]
# #设置画布
# fig, ax = plt.subplots()
# plt.plot(range(w1), REAL, color='#000000',linewidth = '1', marker = '>' ,label='Acutual value')
# plt.plot(range(w1), SAE, color='#4b6cff', linewidth = '1',marker = 's',label='SAE')
# plt.plot(range(w1), SEAE, color='#FFBD23', linewidth = '1',marker = 'p',label='SEAE')
# # plt.plot(range(w1), SGTAE, color='#16c41c', linewidth = '2', label='GSTAE')
# # plt.plot(range(w1), SISAE_GRU, color='#C71585',linewidth = '2', label='SISAE-GRU')
# plt.plot(range(w1), MDTSGAE, color='R',linewidth = '1',marker = '*',  label='FE-SFAE')
# # 设置xtick和ytick的方向：in、out、inout
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.tick_params(axis='both',which='major',labelsize=25)
# plt.xlabel('样本数量',fontsize=20)
# plt.ylabel('预测齐聚物密度值',fontsize=20)
# #plt.title('Quality variable prediction curves',fontsize=25)
# y_major_locator=MultipleLocator(0.1)
# #把y轴的刻度间隔设置为10，并存在变量里
# #ax=plt.gca()
# plt.ylim(0.20,0.82)
# #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
# # plt.tick_params(axis='y', rotation=0 )
# plt.tick_params(axis='both',which='major',labelsize=20)
# axins = ax.inset_axes((0.45, 0.1, 0.2, 0.2))
# axins.plot(range(w1), REAL, color='#000000',marker = '>' ,linewidth = '1',  label='Acutual value')
# axins.plot(range(w1), SAE, color='#4b6cff',marker = 's' , linewidth = '1',label='SAE')
# axins.plot(range(w1), SEAE, color='#FFBD23',marker = 'p' , linewidth = '1',label='SEAE')
# # axins.plot(range(w1), SGTAE, color='#16c41c', linewidth = '2', label='GSTAE')
# # axins.plot(range(w1), SISAE_GRU, color='#C71585',linewidth = '2', label='SISAE-GRU')
# axins.plot(range(w1), MDTSGAE, color='R',linewidth = '1',marker = '*' ,  label='FE-SFAE')
# # 设置放大区间
# zone_left = 100
# zone_right = 150
# # Y轴的显示范围
# y = np.hstack((SAE[zone_left:zone_right], SAE[zone_left:zone_right],
#                SAE[zone_left:zone_right],SAE[zone_left:zone_right],
#                SAE[zone_left:zone_right],SAE[zone_left:zone_right]))
# # 调整子坐标系的显示范围
# axins.set_xlim(160, 180)
# axins.set_ylim(0.25, 0.4)
# # zone_and_linked(ax, axins, 160, 180, range(w1) , [REAL,SAE,SEAE,SGTAE,SISAE_GRU,MDTSGAE], 'right')
# zone_and_linked(ax, axins, 160, 180, range(w1) , [REAL,SAE,SEAE,MDTSGAE], 'right')
# plt.legend(fontsize=20,bbox_to_anchor=(0.7,0.6))
# plt.show()
# #plt.savefig("predict.pdf",dpi=800)


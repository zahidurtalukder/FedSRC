
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from utils.functions_new import save_file, open_file

plt.rcParams["font.family"] = 'sans serif'
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= 'fedavg femnist noniid.pkl'
inf= open_file(data1)

data2= 'fedsrc femnist noniid 3.pkl'
alpha_3= open_file(data2)

data3= 'fedsrc femnist noniid 2.5.pkl'
alpha_2_5= open_file(data3)

data4='fedsrc femnist noniid 2.pkl'
alpha_2= open_file(data4)

data5= 'fedsrc femnist noniid 1.5.pkl'
alpha_1_5= open_file(data5)

data6='fedsrc femnist noniid.pkl'
alpha_1= open_file(data6)

data7= 'fedsrc femnist noniid 0.5.pkl'
alpha_0_5= open_file(data7)

data8= 'fedsrc femnist noniid 0.1.pkl'
alpha_0_1= open_file(data8)

# alpha=[inf[0][-1]*100,alpha_3[0][-1]*100,alpha_2_5[0][-1]*100,alpha_2[0][-1]*100,alpha_1_5[0][-1]*100,alpha_1[0][-1]*100,
#        alpha_0_5[0][-1]*100,alpha_0_5[0][-1]*100]
alpha=[alpha_0_5[1][-1],alpha_0_5[1][-1],alpha_1[1][-1],alpha_1_5[1][-1],alpha_2[1][-1],alpha_2_5[1][-1],
       alpha_3[1][-1],inf[1][-1]]

plt.figure(figsize=(10,8))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)

plt.plot(alpha, '-o', c='blue', linewidth=10.0, zorder=5, marker='^',markevery=1,markersize=25, mfc='white', mec='blue', mew=1,label='FedSRC')

my_xticks = ['0.1','0.5','1','1.5','2','2.5',"3","Inf"]
x = np.array([0,1,2,3,4,5,6,7])
plt.xticks(x, my_xticks,fontsize=35)
plt.xlabel("Alpha Value", weight='bold')
plt.ylabel('Loss', weight='bold')
# plt.yticks(np.arange(40, 101, 15))
# plt.ylim(40, 100)
plt.xlim(-1, 8)
plt.legend(loc=0,  handlelength=1,fontsize=32, ncol=2,
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)



plt.show
plt.savefig('../../figure/effect of alpha loss.png', bbox_inches='tight')


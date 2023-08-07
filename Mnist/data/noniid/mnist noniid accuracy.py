
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from utils.functions_new import save_file, open_file

plt.rcParams["font.family"] = 'sans serif'
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= 'clean Mnist noniid.pkl'
clean= open_file(data1)

data2= 'fedasl Mnist noniid.pkl'
fedasl= open_file(data2)

data3= 'fedavg Mnist noniid.pkl'
fedavg= open_file(data3)

data4='fedsrc Mnist noniid.pkl'
fedsrc= open_file(data4)

data5= 'krum Mnist noniid.pkl'
krum= open_file(data5)

data6='median Mnist noniid.pkl'
median= open_file(data6)

data7= 'tm Mnist noniid.pkl'
tm= open_file(data7)

plt.figure(figsize=(10,8))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)

plt.plot(np.array(clean[0])*100, '--', c='red', linewidth=3.0, zorder=4,  mfc='red', mec='red', mew=1, label='Clean')
plt.plot(np.array(fedasl[0])*100, '--', c='green', linewidth=3.0, zorder=5,  mfc='green', mec='green', mew=1, label='FedASL')
plt.plot(np.array(median[0])*100, ':', c='magenta', linewidth=3.0, zorder=2,  mfc='magenta', mec='magenta', mew=1,label='Median')
plt.plot(np.array(krum[0])*100, '-.', c='cyan', linewidth=3.0, zorder=3,  mfc='cyan', mec='cyan', mew=1,label='Krum')
plt.plot(np.array(tm[0])*100, ':', c='black', linewidth=3.0, zorder=2,  mfc='black', mec='magenta', mew=1,label='TM')
plt.plot(np.array(fedavg[0])*100, '-.', c='darkorange', linewidth=3.0, zorder=3,  mfc='orange', mec='orange', mew=1,label='FedAvg')
plt.plot(np.array(fedsrc[0])*100, '-', c='blue', linewidth=3.0, zorder=6,  mfc='blue', mec='blue', mew=1,label='FedSRC')



plt.xticks(np.arange(0.0, 301, 100))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy (%)', weight='bold')
# plt.yticks(np.arange(40, 101, 15))
# plt.ylim(40, 100)
plt.xlim(0, 300)
plt.legend(loc=0,  handlelength=1,fontsize=32, ncol=2,
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)



plt.show
plt.savefig('../../figure/mnist noniid accuracy.png', bbox_inches='tight')


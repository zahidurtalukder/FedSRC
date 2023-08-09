
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from utils.functions_new import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= 'clean Mnist iid.pkl'
clean= open_file(data1)

data2= 'fedasl Mnist iid.pkl'
fedasl= open_file(data2)

data3= 'fedavg Mnist iid.pkl'
fedavg= open_file(data3)

data4='fedsrc Mnist iid.pkl'
fedsrc= open_file(data4)

data5= 'krum Mnist iid.pkl'
krum= open_file(data5)

data6='median Mnist iid.pkl'
median= open_file(data6)

data7= 'tm Mnist iid.pkl'
tm= open_file(data7)

plt.figure(figsize=(10, 6))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)

# plt.plot(clean[1], '-o', c='royalblue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,label='clean')
plt.plot(fedasl[1], '-o', c='red', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='red', mew=1,label='FedASL')
plt.plot(fedavg[1], '-o', c='green', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='green', mew=1,label='FedAVG')
plt.plot(fedsrc[1], '-o', c='blue', linewidth=3.0, zorder=1, markersize=1, mfc='w', mec='brown', mew=1,label='FedSRC')
plt.plot(krum[1], '-o', c='orange', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='brown', mew=1,label='Krum')
# plt.plot(median[1], '-o', c='cyan', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='brown', mew=1,label='Median')
plt.plot(tm[1], '-o', c='magenta', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='brown', mew=1,label='Trimmed mean')


plt.legend(loc=4)
plt.xticks(np.arange(0.0, 301, 50))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.legend(fontsize=30, loc="upper left",
                 mode="expand", ncol=2)

plt.show
# plt.savefig('../../Figures/mnist/mnist_flipping.pdf', bbox_inches='tight')





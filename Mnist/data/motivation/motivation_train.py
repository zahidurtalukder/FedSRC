
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from utils.functions_new import save_file, open_file
import seaborn as sns

plt.rcParams["font.family"] = 'sans serif'
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

file_name="fedavg Mnist noniid_2.pkl"
fedavg=open_file(file_name)

noise=fedavg[-2]+200
shuffle=fedavg[-3]+100
flip=fedavg[-4]

# loss_test=fedavg[3][-1]
loss_train=fedavg[3][-1]
median1 = np.median(loss_train)
noiseplot=[loss_train[i] for i in noise]
flipplot=[loss_train[i] for i in flip]
shuffleplot=[loss_train[i] for i in shuffle]


result_list = []
result_list.extend(shuffle)
result_list.extend(flip)
result_list.extend(noise)
goodplot = [loss_train[i] for i in range(len(loss_train)) if i not in result_list]




plt.figure(figsize=(10,8))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)

sns.histplot(noiseplot, color='blue', label='Nosiy', kde=True)
sns.histplot(flipplot, color='orange', label='Flipping', kde=True)
sns.histplot(shuffleplot, color='green', label='Shuffling', kde=True)
sns.histplot(goodplot, color='red', label='Good', kde=True)

plt.axvline(x=median1, color='black', linestyle='dashed', linewidth=5, label='Median(all)')

plt.xlabel('Train Loss Value',weight='bold')
plt.ylabel('Frequency',weight='bold')
plt.legend(loc=0,  handlelength=1,fontsize=25, ncol=1,
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)



plt.show
# plt.savefig('../../figure/motivation_train.png', bbox_inches='tight')

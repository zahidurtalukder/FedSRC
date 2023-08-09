import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.functions_new import save_file, open_file
import pandas as pd
# Set Seaborn style

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 35,'font.weight':'bold','pdf.fonttype':42})

# sns.set(style="whitegrid")

data1= 'fedavg femnist noniid.pkl'
avg= open_file(data1)

data2= 'fedsrc femnist noniid.pkl'
avg_src= open_file(data2)

data3= 'fedasl femnist noniid.pkl'
asl= open_file(data3)

data4='fedsrc fedasl femnist noniid.pkl'
asl_src= open_file(data4)

data5= 'krum femnist noniid.pkl'
krum= open_file(data5)

data6='fedsrc krum femnist noniid.pkl'
krum_src= open_file(data6)

data7= 'median femnist noniid.pkl'
median= open_file(data7)

data8= 'fedsrc median femnist noniid.pkl'
median_src= open_file(data8)

data9='tm femnist noniid.pkl'
tm= open_file(data9)

data10= 'fedsrc tm femnist noniid.pkl'
tm_src= open_file(data10)


# Replace these with your actual calculated loss values
algorithm1_without_FedSRC = avg[0][-1]*100
algorithm1_with_FedSRC = avg_src[0][-1]*100

algorithm2_without_FedSRC = asl[0][-1]*100
algorithm2_with_FedSRC = asl_src[0][-1]*100

algorithm3_without_FedSRC = krum[0][-1]*100
algorithm3_with_FedSRC = krum_src[0][-1]*100

algorithm4_without_FedSRC = median[0][-1]*100
algorithm4_with_FedSRC = median_src[0][-1]*100

algorithm5_without_FedSRC = tm[0][-1]*100
algorithm5_with_FedSRC = tm_src[0][-1]*100



# List of algorithms and their corresponding loss values
algorithms = ['FedAVG', 'FedASL', 'Krum', 'Median', 'TM']
without_FedSRC = [algorithm1_without_FedSRC, algorithm2_without_FedSRC, algorithm3_without_FedSRC, algorithm4_without_FedSRC, algorithm5_without_FedSRC]
with_FedSRC = [algorithm1_with_FedSRC, algorithm2_with_FedSRC, algorithm3_with_FedSRC, algorithm4_with_FedSRC, algorithm5_with_FedSRC]

# Create a DataFrame for the data
data = {
    'Algorithms': algorithms * 2,
    'Accuracy': without_FedSRC + with_FedSRC,
    'FedSRC': ['Without FedSRC'] * len(algorithms) + ['With FedSRC'] * len(algorithms)
}
df = pd.DataFrame(data)

# Create the bar plot using Seaborn
plt.figure(figsize=(10, 8))
sns.barplot(x='Algorithms', y='Accuracy', hue='FedSRC', data=df, palette=['red', 'blue'])
# plt.xlabel('Algorithms', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.xticks(fontsize=28)
plt.yticks(np.arange(0, 101, 20))
plt.ylim(0, 90)
plt.legend(loc=0, ncol=2, fontsize=28, borderaxespad=0.02, handlelength=0.7, framealpha=0.5, handletextpad=0.2, labelspacing=0.2, columnspacing=0.6)
plt.tight_layout()
plt.show()

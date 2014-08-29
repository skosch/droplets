import numpy as np
import pandas as pd
import matplotlib as mpl
sizes = [127, 219, 299, 361, 384, 447]
letters = ["A", "B", "C", "D", "E", "F"]

data = pd.DataFrame()
for size, letter in zip(sizes, letters):
    with open("slit" + str(size), "r") as f:
        data[letter] = pd.Series( np.array(f.readlines()[1].split(", ")).astype(np.float) )

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font="Alte DIN 1451 Mittelschrift")
mpl.rc("xtick", labelsize=16)

f = plt.figure(figsize=(4.5, 3.5))

for i, (size, letter) in enumerate(zip(sizes, letters)):
    print data[letter].mean()
    sp = plt.subplot("71" + str(i+1))
    plt.yticks(visible=False)
    sp.set_ylabel("F" + letter, fontsize="16", rotation=270, color="w", labelpad=9)
    sns.distplot(data[letter], bins=np.arange(4, 39, (39.0-4.0)/130.0),
            kde=True, color="k", axlabel=False, kde_kws={"kernel": "gau", "bw":
                0.5})
    sp.spines['left'].set_linewidth(18)
    sp.spines['left'].set_edgecolor('0.5')
    sp.spines['left'].set_zorder(-10)
    sp.yaxis.grid(False)
    plt.xlim(4, 39)
    if i < 5:
        plt.setp(sp.get_xticklabels(), visible=False)
plt.show()

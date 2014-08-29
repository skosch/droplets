# coding=utf8
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font="Alte DIN 1451 Mittelschrift")
mpl.rc("xtick", labelsize=13)
mpl.rc("ytick", labelsize=13)
sns.set_palette("binary")
xvals = [127, 220, 299, 361, 384, 447]
yvals = [9.71, 16.7, 22.92, 27.26, 29.89, 34.56]

data = pd.DataFrame()
data['x'] = pd.Series(xvals)
data['y'] = pd.Series(yvals)
sp = sns.regplot("x", "y", data, scatter_kws={"color": "k",
    "zorder": 100, "s": 25}, line_kws={"color": "0.5"}, label="0.07812")
sp.set_xlabel("Diameter [um]", fontsize="16")
sp.set_ylabel("Fringes", fontsize="16")
#sp.plot([535], [31.72], color="r", markersize=6, marker="D", fillstyle="none", markeredgewidth=1, markeredgecolor="r", zorder=104)
plt.show()

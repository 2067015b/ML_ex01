from matplotlib import pyplot as plt
import numpy as np
import math

from ML_ex01.Classification.utils import normalize_data

X = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]

X, X_test = normalize_data(X,X_test)

data = {0:[],1:[]}
for i in range(y.shape[0]):
    if y[i] == 2:
        data[1].append(X[i,:])
    else:
        data[0].append(X[i,:])

data[0] = np.array(data[0])
data[1] = np.array(data[1])


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,36))

for ax,cnt in zip(axes.ravel(), range(100,104)):

    # set bin sizes
    min_b = math.floor(np.min(X[:,cnt]))
    max_b = math.ceil(np.max(X[:,cnt]))
    bins = np.linspace(min_b, max_b, 25)

    # plottling the histograms
    for lab,col in zip(range(0,2), ('blue', 'red')):
        ax.hist(data[lab][:,cnt],
                   color=col,
                   label='class %s' % str(lab),
                   bins=bins,
                   alpha=0.5,)
    ylims = ax.get_ylim()

    # plot annotation
    leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)
    ax.set_ylim([0, max(ylims)+2])
    ax.set_xlabel(str(cnt))
    #ax.set_title('Iris histogram #%s' %str(cnt+1))

    # hide axis ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

axes[0][0].set_ylabel('count')
axes[1][0].set_ylabel('count')

# fig.tight_layout()
plt.show()

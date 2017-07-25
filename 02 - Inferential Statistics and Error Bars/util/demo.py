import time
import numpy as np
import matplotlib.pyplot as plt


def setupRun(pmf,iters):
    pmfs = [pmf]
    xMax = iters*(len(pmf))
    xLocations = list(range(xMax+2))
    xLabels = [str(loc) if (loc%(len(pmf)-1))==0 else '' for loc in xLocations]
    extendedPMF = np.hstack([pmfs[0],[0]*(xMax+2-len(pmfs[0]))])
    edge = 2
    fig = plt.figure(figsize=(12,6)); pmfAx = plt.subplot(111)
    pmfBars = pmfAx.bar(xLocations,extendedPMF,width=1,align='center',alpha=0.8,
                       linewidth=0,)

    setupPlot(plt.gca(),xLocations,edge,xLabels)

    plt.suptitle("Adding Up "+str(iters)+" Random Numbers",
             size=24,weight='bold',y=1.);
    fig.canvas.draw()

    return fig,pmfBars,pmfs

def setupPlot(ax,locs,edge,labels):
    ax.set_ylim([0,1]); ax.set_xlim([locs[0]-edge,locs[1]+edge]);
    ax.xaxis.set_ticks(locs); ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticks([0,0.5,1]);
    ax.tick_params(axis='x',top='off')
    ax.tick_params(axis='y',right='off')
    plt.ylabel('Probability',fontsize='x-large',fontweight='bold')

def centralLimitDemo(pmf,iters):
    assert min(pmf) >= 0, "no negative numbers in pmf"
    assert np.isclose(sum(pmf), 1), "doesn't sum to 1"
    assert max(pmf) < 1, "must have non-zero variance"

    figure,barPlot,pmfs = setupRun(pmf,iters)
    time.sleep(0.2)
    for _ in range(iters):
        [barPlot[idx].set_height(h)
             for idx,h in enumerate(pmfs[-1])]
        pmfs.append(np.convolve(pmfs[-1],pmfs[0]))
        figure.canvas.draw()
        time.sleep(0.1*(1-0.1)**_)

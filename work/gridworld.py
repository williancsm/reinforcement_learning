import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython import display
import time
import json
from copy import deepcopy

plt.rc('font', size=30)  # controls default text sizes
plt.rc('axes', titlesize=25)  # fontsize of the axes title
plt.rc('axes', labelsize=25)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=17)  # fontsize of the tick labels
plt.rc('ytick', labelsize=17)  # fontsize of the tick labels
plt.rc('legend', fontsize=20)  # legend fontsize
plt.rc('figure', titlesize=30)
plt.tight_layout()


def plot(V, pi):
    # plot value
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    ax1.axis('on')
    ax1.cla()
    states = np.arange(V.shape[0])
    ax1.bar(states, V, edgecolor='none')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Value', rotation='horizontal', ha='right')
    ax1.set_title('Value Function')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax1.yaxis.grid()
    ax1.set_ylim(bottom=V.min())
    # plot policy
    ax2.axis('on')
    ax2.cla()
    im = ax2.imshow(pi.T, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    ax2.invert_yaxis()
    ax2.set_xlabel('State')
    ax2.set_ylabel('Action', rotation='horizontal', ha='right')
    ax2.set_title('Policy')
    start, end = ax2.get_xlim()
    ax2.xaxis.set_ticks(np.arange(start, end), minor=True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start, end), minor=True)
    ax2.grid(which='minor')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.20)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Probability', rotation=0, ha='left')
    fig.subplots_adjust(wspace=0.5)
    display.clear_output(wait=True)
    display.display(fig)
    time.sleep(0.001)

    plt.savefig('my_figure.png')
    plt.close()


class GridWorld:

    def __init__(self, size, initial_value=None):
        self.size = size
        self.states = np.arange(size * size)
        self.actions = np.arange(4)
        self.grid = np.full((size, size), initial_value)

    def transitions(self, s, a):
        return np.array([[r, self.p(s_, r, s, a)]
                         for s_, r in self.support(s, a)])

    def support(self, s, a):
        return [(s_, self.reward(s, s_)) for s_ in self.states]

    def p(self, next_state, reward, state, action):
        row = state // self.size
        col = state % self.size

        if (state == 0 or state == len(self.states) - 1):
            return 0.0

        #up = 0
        if (action == 0):
            new_col = col
            new_row = max(row - 1, 0)

        #down = 1
        if (action == 1):
            new_col = col
            new_row = min(row + 1, self.size - 1)

        #left = 2
        if (action == 2):
            new_col = max(col - 1, 0)
            new_row = row

        #right = 3
        if (action == 3):
            new_col = min(col + 1, self.size - 1)
            new_row = row

        
        new_state = new_row * self.size + new_col
        if (new_state == next_state):
            return 1.0
        else:
            return 0.0

    def reward(self, state, next_state):        
        if (state == 0 or state == len(self.states) - 1):
            return 0.0
        else:
            return -1.0

    @property
    def A(self):
        return list(self.actions)

    @property
    def S(self):
        return list(self.states)


class Transitions(list):

    def __init__(self, transitions):
        self.__transitions = transitions
        super().__init__(transitions)

    def __repr__(self):
        repr = '{:<14} {:<10} {:<10}'.format('Next State', 'Reward',
                                             'Probability')
        repr += '\n'
        for i, (s, r, p) in enumerate(self.__transitions):
            repr += '{:<14} {:<10} {:<10}'.format(s, round(r, 2), round(p, 2))
            if i != len(self.__transitions) - 1:
                repr += '\n'
        return repr

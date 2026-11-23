"""Reproducible PNG figures for the CPU experiment report."""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogFormatter
import numpy as np

from report_data import HERE, REAL_CASES, STUDY, read, timing, timing_runs


BLUE = '#246B8B'
ORANGE = '#B45736'
GRAY = '#313C44'
mpl.rcParams.update({
    'font.family': ['Arial', 'PingFang SC', 'Heiti SC'],
    'font.size': 9.3,
    'mathtext.fontset': 'dejavusans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#53636D',
    'axes.labelcolor': '#152832',
    'xtick.color': '#152832',
    'ytick.color': '#152832',
    'axes.unicode_minus': True,
})


def save(fig, folder, filename):
    fig.savefig(folder / filename, dpi=340, facecolor='white')
    plt.close(fig)


def two_pore(folder):
    data = read('two_pore.json')
    exact = data['exact']
    coarse = data['runs'][-1]['modes']['reference_only']
    fig, axs = plt.subplots(1, 2, figsize=(6.82, 2.20), gridspec_kw={'width_ratios': [1.05, 1]})
    fig.subplots_adjust(left=.095, right=.975, bottom=.25, top=.77, wspace=.38)
    n = np.arange(3)
    axs[0].plot(n, exact['P_N'], 'o-', color=BLUE, lw=1.8, ms=5)
    axs[0].plot(n, coarse['P_N'], 's--', color=ORANGE, lw=1.1, ms=7,
                markerfacecolor='none', markeredgewidth=1.15)
    axs[0].set(xlabel='总分子数 N', ylabel='概率 P(N)', xticks=n, ylim=(0, .72),
               yticks=[0, .2, .4, .6])
    for k, (item, color) in enumerate(((exact, BLUE), (coarse, ORANGE))):
        axs[1].bar(np.arange(2) + (k - .5) * .31, item['mean_pore_occupancy'],
                   width=.27, color=color)
    axs[1].set(ylabel='占据概率', xticks=[0, 1], xticklabels=['A 孔', 'B 孔'],
               ylim=(0, 1.00), yticks=[0, .25, .5, .75, 1])
    for i, axis in enumerate(axs):
        axis.text(-.11, 1.07, 'ab'[i], transform=axis.transAxes, family='Arial', weight='bold', fontsize=11)
        axis.grid(axis='y', color='#E5EAED', lw=.6)
        axis.set_axisbelow(True)
    fig.legend([Line2D([], [], color=BLUE, marker='o'), Line2D([], [], color=ORANGE, marker='s',
                markerfacecolor='none', ls='--')], ['枚举真值', '最粗参考势'],
               loc='upper center', bbox_to_anchor=(.53, 1.002), ncol=2, frameon=False)
    save(fig, folder, 'double_pore.png')


def nist_distribution(folder):
    pure = 'nist_co2_zif8'
    mix = 'nist_n2_co2_zif8'
    fig, axs = plt.subplots(1, 2, figsize=(6.82, 2.55), gridspec_kw={'width_ratios': [1.13, 1]})
    fig.subplots_adjust(left=.095, right=.93, bottom=.25, top=.78, wspace=.42)
    with np.load(STUDY / 'inputs' / f'{pure}_nist.npz') as ref:
        x = ref['counts'][:, 0]
        p = ref['probabilities'].mean(axis=0)
        axs[0].plot(x, p, color=GRAY, lw=1.6, label='NIST TMMC')
    for mode, color, marker, label in (
        ('reference-mh', BLUE, 'o', '单阶段'),
        ('delayed', ORANGE, 's', '两阶段'),
    ):
        pairs = np.array(read(f'm5/{pure}-{mode}.json')['occupancy_distribution'])
        axs[0].plot(pairs[:, 0], pairs[:, 1], color=color, marker=marker, ms=3.4,
                    ls='none', markerfacecolor='none', label=label)
    axs[0].set(xlim=(0, 20), xticks=range(0, 21, 5), ylim=(0, .20),
               xlabel='CO₂ 分子数', ylabel='概率 P(N)')
    axs[0].grid(axis='y', color='#E5EAED', lw=.6)
    ax = axs[1]
    with np.load(STUDY / 'inputs' / f'{mix}_nist.npz') as ref:
        n = ref['counts']
        p = ref['probabilities'].mean(axis=0)
    valid = (n[:, 0] <= 9) & (n[:, 1] <= 30)
    dense = np.zeros((31, 10))
    dense[n[valid, 1], n[valid, 0]] = p[valid]
    raster = ax.imshow(dense, origin='lower', aspect='auto', interpolation='nearest',
                       cmap='Blues', extent=(-.5, 9.5, -.5, 30.5), vmin=0)
    ax.plot([1.5, 1.5, 9.5], [30.5, 12.5, 12.5], color=ORANGE, lw=1.25, ls='--')
    ax.set(xlabel='N₂ 分子数', ylabel='CO₂ 分子数', xticks=[0, 2, 4, 6, 8],
           yticks=[0, 10, 20, 30])
    cbar = fig.colorbar(raster, ax=ax, shrink=.94, pad=.035, fraction=.095)
    cbar.ax.tick_params(labelsize=8, length=2)
    cbar.locator = mpl.ticker.MaxNLocator(3)
    cbar.update_ticks()
    for k, axis in enumerate(axs):
        axis.text(-.11, 1.08, 'ab'[k], transform=axis.transAxes, family='Arial', weight='bold', fontsize=11)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.48, .985),
               ncol=3, columnspacing=1.1, frameon=False)
    save(fig, folder, 'nist.png')


def paired_timing(folder):
    fig, axs = plt.subplots(1, 2, figsize=(6.82, 3.45), sharey=True)
    fig.subplots_adjust(left=.254, right=.97, bottom=.18, top=.77, wspace=.14)
    y = np.arange(len(REAL_CASES))
    for axis, side, name in zip(axs, ('m5', 'hpc'), ('Apple M5', 'HPC Intel Xeon')):
        for row, case in enumerate(REAL_CASES):
            native, raspa = timing_runs(side, case)
            for offset, color, marker, record in (
                (-.16, BLUE, 'o', native), (.16, ORANGE, 's', raspa),
            ):
                value, censored = timing(record)
                yy = row + offset
                if censored:
                    axis.annotate('', xy=(value*1.34, yy), xytext=(value, yy),
                                  arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': 1.25,
                                              'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': 8})
                axis.plot(value, yy, marker=marker, ls='none', color=color, ms=4.3,
                          markerfacecolor='white' if censored else color, zorder=5)
        axis.set_xscale('log')
        axis.set(xlim=(.75, 2800), ylim=(len(y)-.5, -.5), title=name)
        axis.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        axis.xaxis.set_major_formatter(LogFormatter(labelOnlyBase=True))
        axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        axis.grid(axis='x', color='#DDE6EA', lw=.7)
        axis.tick_params(axis='y', length=0, pad=8)
        axis.set_xlabel('首达标墙钟时间 / s')
    axs[0].set_yticks(y, [x[0] for x in REAL_CASES], fontsize=9)
    fig.legend([Line2D([], [], color=BLUE, marker='o', linestyle='none'),
                Line2D([], [], color=ORANGE, marker='s', linestyle='none')],
               ['OpenGCMC 两阶段', 'RASPA3 CBMC'], loc='upper center',
               bbox_to_anchor=(.64, .99), ncol=2, frameon=False)
    save(fig, folder, 'timing.png')


if __name__ == '__main__':
    folder = HERE / 'figures'
    folder.mkdir(exist_ok=True)
    for chart in (two_pore, nist_distribution, paired_timing):
        chart(folder)

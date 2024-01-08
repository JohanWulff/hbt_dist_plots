from collections import OrderedDict

from pathlib2 import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import uproot
from hist import Hist
import mplhep as hep

luminosities = {  # in /fb
    2016: 36.310,
    2017: 41.480,
    2018: 59.830,
}

def hist_stackplot(n_stack: {},
                   n_signal: Hist,
                   limit_val,
                   savepath="",
                   ):
    hep.histplot([val for key, val in n_stack.items()],
                stack=True,
                histtype='step',
                yerr=True,
                label=[key for key, val in n_stack.items()])
    norm_sig = (n_signal/n_signal.sum().value)*limit_val
    hep.histplot(norm_sig, label=f'signal normalized to limit: {norm_sig.sum().value:.1f} [fb]',color='black')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.yscale("log")
    plt.ylabel("N")
    plt.xlabel("DNN out")
    if savepath != "":
        plt.savefig(f"{savepath}", bbox_inches='tight')
    plt.close()


def plot_hist(filename: str,
         exp_lim: np.array,
         masses: np.array, 
         savedir: str):
    filename = Path(filename)
    _, _, year, channel, cat, _, spin, _, mass = filename.stem.split("_")
    dirname = f"cat_{year}_{channel}_{cat}"
    signal_name = f"ggf_spin_{spin}_mass_{mass}_hbbhtt"

    limit = exp_lim[np.where(masses==float(mass))]
    limit_val = limit[0]

    n_stack = {}
    with uproot.open(filename) as file:
        objects = file[dirname].classnames()
        for sample in objects:
            if not any(i in sample for i in ['ggf', 'data_obs']):
                n_stack[sample.strip(";1")] = file[dirname][sample].to_hist()
            n_signal = file[dirname][signal_name].to_hist()
    hist_stackplot(n_stack,
                   n_signal,
                   limit_val,
                   savepath=f"{savedir}/{dirname}_{signal_name}.pdf")


def plot_stacked_hist(n_samples: dict,
                    n_signal: tuple,
                    lumi_scale: float,
                    limit_val: float,
                    merge={},
                    xlabel="DNN out",
                    ylabel="N",
                    title="",
                    savepath="",
                    log=True):
    #fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios':[3,1]})
    fig, axs = plt.subplots(1,1)
    bottom = np.zeros(shape=n_signal[0].shape)
    centers = (n_signal[1][1:]+n_signal[1][:-1])/2.
    widths = n_signal[1][1:]-n_signal[1][:-1]
    n_sig = n_signal[0]
    if bool(merge):
        for i in merge:
            sum_list = [n_samples[m][0] for m in merge[i]]
            n_samples[i] = (np.sum(sum_list, axis=0), n_samples[list(n_samples.keys())[0]][1]) 
            for m in merge[i]:
                del n_samples[m]
    n_samples = {key: value for key, value in sorted(n_samples.items(), key=lambda x: np.sum(x[1][0]), reverse=True)}
    yields = {sample : lumi_scale*np.sum(n_samples[sample][0]) for sample in n_samples} 
    for sample in n_samples:
        axs.bar(centers,
                lumi_scale*n_samples[sample][0],
                width=widths,
                bottom=bottom,
                align='center',
                label=f"{sample} N: {yields[sample]:.2f}")
        bottom += lumi_scale*n_samples[sample][0]
    n_sig /= np.sum(n_sig)
    n_sig *= limit_val
    axs.bar(centers,n_signal[0], width=widths, fill=None, 
            edgecolor='black', label=f'signal norm. to limit: {np.sum(n_signal[0]):.2f}')
    #axs.step(n_signal[1], n_signal[0], width=widths, align='center',
            #fill=None,edgecolor='black', label=f'signal N: {np.sum(n_signal[0]):.2f}')
    axs.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    if log == True:
        axs.set_yscale('log')
    if title != "":
        axs.set_title(title)
    axs.set_ylabel(ylabel)
    axs.set_xlim([n_signal[1][0], n_signal[1][-1]])
    axs.set_xlabel(xlabel)
    if savepath != "":
        plt.tight_layout()
        plt.savefig(f"{savepath}")
    plt.close()


def plot(filename, exp_lim, masses, savedir):
    filename = Path(filename)
    _, _, year, channel, cat, _, spin, _, mass = filename.stem.split("_")
    dirname = f"cat_{year}_{channel}_{cat}"
    signal_name = f"ggf_spin_{spin}_mass_{mass}_hbbhtt"

    limit = exp_lim[np.where(masses==float(mass))]
    limit_val = limit[0]
    with uproot.open(filename) as file:
        objects = file[dirname].classnames()
        n_stack = {}
        for sample in objects:
            if not any(i in sample for i in ['ggf', 'data_obs']):
                n_stack[sample.strip(";1")] = file[dirname][sample].to_numpy()
        n_signal = file[dirname][signal_name].to_numpy()
    merge = {'VV': ['WW','WZ','ZZ'], 'SM_H': ['ttH_hbb', 'ttH_htt', 'ggH_htt', 'ZH_htt', 'WH_htt', 'qqH_htt']}
    plot_stacked_hist(n_samples=n_stack,
                        n_signal=n_signal,
                        lumi_scale=1,# luminosities[int(year)],
                        limit_val=limit_val,
                        merge=merge,
                        title=f'{" ".join(dirname.split("_")[1:])} {" ".join(signal_name.split("_"))}',
                        savepath=f"{savedir}/{dirname}_{signal_name}.pdf")
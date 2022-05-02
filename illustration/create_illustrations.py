"""
Illustration script for low rank paper
Author: Steffen Schotthöfer, Jonas Kusch
Date: 22.04.2022
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # 1) Illustrate training performance of 3 way low rank, 1 way low rank, and full rank
    folder = "paper_data/"
    dlra_3layer = pd.read_csv(folder + "3LayerDLRA_MNIST.csv", delimiter=";", header=None,
                              names=["train_loss", "train_acc", "vall_loss", "vall_acc", "vall_loss_trash", "rank1",
                                     "rank2", "rank3"])

    plt.plot(dlra_3layer[["train_loss", "train_acc", "vall_loss", "vall_acc"]])
    plt.savefig("figures/dlra_3_layer.png")
    plt.clf()
    plt.plot(dlra_3layer[["rank1", "rank2", "rank3"]])
    plt.savefig("figures/dlra_3_layer_rank.png")
    plt.clf()

    # 2) Create plots for parameter study
    folder = "paper_data/sr_100/200x3_sr100_v0.01/historyLogs"
    dlra_3layer = pd.read_csv(folder + "/history_001_.csv", delimiter=";", header=None,
                              names=["train_loss", "train_acc", "vall_loss", "vall_acc", "vall_loss_trash", "rank1",
                                     "rank2", "rank3"])

    plt.plot(dlra_3layer[["train_loss", "train_acc", "vall_loss", "vall_acc"]])
    plt.savefig("figures/sr100_01.png")
    plt.clf()
    plt.plot(dlra_3layer[["rank1", "rank2", "rank3"]])
    plt.savefig("figures/sr100_01_rank.png")
    plt.clf()

    folder = "paper_data/sr_100/200x3_sr100_v0.03/historyLogs"
    dlra_3layer = pd.read_csv(folder + "/history_001_.csv", delimiter=",", header=None,
                              names=["train_loss", "train_acc", "vall_loss", "vall_acc", "vall_loss_trash", "rank1",
                                     "rank2", "rank3"])

    plt.plot(dlra_3layer[["train_loss", "train_acc", "vall_loss", "vall_acc"]])
    plt.savefig("figures/sr100_03.png")
    plt.clf()
    plt.plot(dlra_3layer[["rank1", "rank2", "rank3"]])
    plt.savefig("figures/sr100_03_rank.png")
    plt.clf()

    folder = "paper_data/sr_100/200x3_sr100_v0.05/historyLogs"
    dlra_3layer = pd.read_csv(folder + "/history_001_.csv", delimiter=",", header=None,
                              names=["train_loss", "train_acc", "vall_loss", "vall_acc", "vall_loss_trash", "rank1",
                                     "rank2", "rank3"])

    plt.plot(dlra_3layer[["vall_loss", "vall_acc"]])
    plt.plot(dlra_3layer[["train_loss"]], '-.')
    plt.plot(dlra_3layer[["train_acc"]], '-.')
    plt.savefig("figures/sr100_05.png")
    plt.clf()
    plt.plot(dlra_3layer[["rank1", "rank2", "rank3"]])
    plt.savefig("figures/sr100_05_rank.png")
    plt.clf()

    folder = "paper_data/sr_100/200x3_sr100_v0.07/historyLogs"
    dlra_3layer = pd.read_csv(folder + "/history_001_.csv", delimiter=",", header=None,
                              names=["train_loss", "train_acc", "vall_loss", "vall_acc", "vall_loss_trash", "rank1",
                                     "rank2", "rank3"])

    plt.plot(dlra_3layer[["vall_loss", "vall_acc"]])
    plt.plot(dlra_3layer[["train_loss"]], '-.')
    plt.plot(dlra_3layer[["train_acc"]], '-.')
    plt.legend(["vall_loss", "vall_acc", "train_loss", "train_acc"])
    plt.savefig("figures/sr100_07.png")
    plt.clf()
    plt.plot(dlra_3layer[["rank1", "rank2", "rank3"]])
    plt.legend(["layer1", "layer2", "layer3"])
    plt.savefig("figures/sr100_07_rank.png")
    plt.clf()

    # 3) Long time runs

    folder = "paper_data/long_time_results/200x3_sr100_v0.07/historyLogs"
    dlra_3layer = pd.read_csv(folder + "/history_004_.csv", delimiter=";")

    plt.plot(dlra_3layer[["acc_val"]], '-')
    plt.plot(dlra_3layer[["acc_train"]], '-.')
    plt.plot(dlra_3layer[["acc_test"]], '--')
    plt.legend(["acc_val", "acc_train", "acc_test"])
    plt.ylim([0.8, 1.05])
    plt.savefig("figures/long_time2000_acc.png", dpi=600)
    plt.clf()

    plt.plot(dlra_3layer[["loss_val"]], '-')
    plt.plot(dlra_3layer[["loss_train"]], '-.')
    plt.plot(dlra_3layer[["loss_test"]], '--')
    plt.legend(["loss_val", "loss_train", "loss_test"])
    plt.ylim([1e-7, 1.01])
    plt.savefig("figures/long_time2000_loss.png", dpi=600)
    plt.yscale('log')
    plt.savefig("figures/long_time2000_loss_log.png", dpi=600)
    plt.clf()
    plt.plot(dlra_3layer[["rank1", "rank2", "rank3"]])
    plt.legend(["layer1", "layer2", "layer3"])
    plt.savefig("figures/long_time2000_ranks.png", dpi=600)
    plt.yscale('log')
    plt.savefig("figures/long_time2000_ranks_log.png", dpi=600)
    plt.clf()

    folder = "paper_data/long_time_results/small_validation_set/200x3_sr100_v0.07/historyLogs"
    dlra_3layer = pd.read_csv(folder + "/history_001_.csv", delimiter=";", header=None,
                              names=["train_loss", "train_acc", "vall_loss", "vall_acc", "vall_loss_trash", "rank1",
                                     "rank2", "rank3"])

    plt.plot(dlra_3layer[["vall_loss", "vall_acc"]])
    plt.plot(dlra_3layer[["train_loss"]], '-.')
    plt.plot(dlra_3layer[["train_acc"]], '-.')
    plt.legend(["vall_loss", "vall_acc", "train_loss", "train_acc"])
    plt.savefig("figures/long_time2000_small_val.png")
    plt.yscale("log")
    plt.savefig("figures/long_time2000_small_val_log.png")
    plt.clf()
    plt.plot(dlra_3layer[["rank1", "rank2", "rank3"]])
    plt.legend(["layer1", "layer2", "layer3"])
    plt.savefig("figures/long_time2000_small_val_ranks.png")
    plt.yscale("log")
    plt.savefig("figures/long_time2000_small_val_ranks_log.png")
    plt.clf()

    return 0


def plot_1d(xs, ys, labels=None, name='defaultName', log=True, folder_name="figures", linetypes=None, show_fig=False,
            xlim=None, ylim=None, xlabel=None, ylabel=None, title: str = r"$h^n$ over ${\mathcal{R}^r}$"):
    plt.clf()
    if not linetypes:
        linetypes = ['-', '--', '-.', ':', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
                     'h',
                     'H',
                     '+', 'x', 'D', 'd', '|']
        if labels is not None:
            linetypes = linetypes[0:len(labels)]

    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 0.7
    if len(xs) == 1:
        x = xs[0]
        for y, lineType in zip(ys, linetypes):
            for i in range(y.shape[1]):
                if colors[i] == 'k' and lineType in ['.', ',', 'o', 'v', '^', '<', '>']:
                    colors[i] = 'w'
                plt.plot(x, y[:, i], colors[i] + lineType, linewidth=symbol_size, markersize=2.5,
                         markeredgewidth=0.5, markeredgecolor='k')
        if labels != None:
            plt.legend(labels)
    elif len(xs) is not len(ys):
        print("Error: List of x entries must be of same length as y entries")
        exit(1)
    else:
        for x, y, lineType in zip(xs, ys, linetypes):
            plt.plot(x, y, lineType, linewidth=symbol_size)
        plt.legend(labels)  # , prop={'size': 6})
    if log:
        plt.yscale('log')

    if show_fig:
        plt.show()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=12)
        # plt.xticks(fontsize=6)
        # plt.yticks(fontsize=6)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.savefig(folder_name + "/" + name + ".png", dpi=400)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".png"))
    return 0


def plot_1dv2(xs, ys, labels=None, name='defaultName', log=True, loglog=False, folder_name="figures", linetypes=None,
              show_fig=False,
              xlim=None, ylim=None, xlabel=None, ylabel=None, title: str = r"$h^n$ over ${\mathcal{R}^r}$"):
    """
    Expected shape for x in xs : (nx,)
                       y in ys : (1,nx)
    """
    plt.clf()
    plt.figure(figsize=(5.8, 4.7), dpi=400)
    if not linetypes:
        linetypes = ['-', '--', '-.', ':', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
                     'h',
                     'H',
                     '+', 'x', 'D', 'd', '|']
        if labels is not None:
            linetypes = linetypes[0:len(labels)]

    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b', 'c', 'm', 'y']
    symbol_size = 0.7
    if len(xs) == 1:
        x = xs[0]
        i = 0
        for y, lineType in zip(ys, linetypes):
            if lineType in ['.', ',', 'o', 'v', '^', '<', '>']:
                if colors[i] == 'k':
                    plt.plot(x, y, 'w' + lineType, linewidth=symbol_size, markersize=2.5,
                             markeredgewidth=0.5, markeredgecolor='k')
                else:
                    plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size, markersize=2.5,
                             markeredgewidth=0.5, markeredgecolor='k')
            else:
                plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size)
            i += 1
        if labels != None:
            plt.legend(labels)
    elif len(xs) is not len(ys):
        print("Error: List of x entries must be of same length as y entries")
        exit(1)
    else:
        for x, y, lineType, color in zip(xs, ys, linetypes, colors):
            plt.plot(x, y, color + lineType, linewidth=symbol_size)
        plt.legend(labels)  # , prop={'size': 6})
    if log:
        plt.yscale('log')
    if loglog:
        plt.yscale('log')
        plt.xscale('log')
    if show_fig:
        plt.show()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=12)
        # plt.xticks(fontsize=6)
        # plt.yticks(fontsize=6)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=12)
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".png", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".png"))
    plt.close()
    return 0


if __name__ == '__main__':
    main()

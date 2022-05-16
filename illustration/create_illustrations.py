"""
Illustration script for low rank paper
Author: Steffen Schotthöfer, Jonas Kusch
Date: 22.04.2022
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    print_param_counts()

    # print_param_cout()
    name = "e2edense_sr300_v0.17"
    plot_run4layer(load_folder="200Neurons/" + name, save_name="200Neurons/" + name)
    name = "e2edense_sr300_v0.15"
    plot_run4layer(load_folder="200Neurons/" + name, save_name="200Neurons/" + name)
    name = "e2edense_sr300_v0.13"
    plot_run4layer(load_folder="200Neurons/" + name, save_name="200Neurons/" + name)
    name = "e2edense_sr300_v0.11"
    plot_run4layer(load_folder="200Neurons/" + name, save_name="200Neurons/" + name)
    name = "e2edense_sr300_v0.09"
    plot_run4layer(load_folder="200Neurons/" + name, save_name="200Neurons/" + name)
    name = "e2edense_sr300_v0.07"
    plot_run4layer(load_folder="200Neurons/" + name, save_name="200Neurons/" + name)
    name = "e2edense_sr300_v0.05"
    plot_run4layer(load_folder="200Neurons/" + name, save_name="200Neurons/" + name)
    name = "e2edense_sr300_v0.03"
    plot_run4layer(load_folder="200Neurons/" + name, save_name="200Neurons/" + name)

    # plot_timings
    plot_timing_exec(load_folder="timings/execution_timings.csv", save_name="timings/execution_timings")
    plot_timing_train(load_folder="timings/fix_rank_training.csv", save_name="timings/fix_rank_training")

    """
    name = "e2edense_sr200_v0.03"
    plot_run4layer(load_folder="4layer/" + name, save_name="4layer/" + name)
    name = "e2edense_sr200_v0.05"
    plot_run4layer(load_folder="4layer/" + name, save_name="4layer/" + name)
    name = "e2edense_sr200_v0.07"
    plot_run4layer(load_folder="4layer/" + name, save_name="4layer/" + name)
    name = "e2edense_sr200_v0.09"
    plot_run4layer(load_folder="4layer/" + name, save_name="4layer/" + name)
    name = "e2edense_sr200_v0.11"
    plot_run4layer(load_folder="4layer/" + name, save_name="4layer/" + name)
    name = "e2edense_sr200_v0.13"
    plot_run4layer(load_folder="4layer/" + name, save_name="4layer/" + name)
    name = "e2edense_sr200_v0.15"
    plot_run4layer(load_folder="4layer/" + name, save_name="4layer/" + name)


    name = "200x3_sr199_v0.07"
    plot_run(load_folder="wednesday/" + name, save_name="wednesday_run/" + name)
    name = "200x3_sr199_v0.05"
    plot_run(load_folder="wednesday/" + name, save_name="wednesday_run/" + name)
    name = "200x3_sr199_v0.03"
    plot_run(load_folder="wednesday/" + name, save_name="wednesday_run/" + name)
    name = "200x3_sr199_v0.01"
    plot_run(load_folder="wednesday/" + name, save_name="wednesday_run/" + name)
    name = "200x3_sr199_v0.005"
    plot_run(load_folder="wednesday/" + name, save_name="wednesday_run/" + name)
    name = "200x3_sr200_v0.07"
    plot_run(load_folder="wednesday/" + name, save_name="wednesday_run/" + name)
    name = "200x3_sr200_v0.05"
    plot_run(load_folder="wednesday/" + name, save_name="wednesday_run/" + name)
    name = "200x3_sr200_v0.03"
    plot_run(load_folder="wednesday/" + name, save_name="wednesday_run/" + name)
    name = "200x3_sr200_v0.01"
    plot_run(load_folder="wednesday/" + name, save_name="wednesday_run/" + name)
    name = "200x3_sr200_v0.005"
    plot_run(load_folder="wednesday/" + name, save_name="wednesday_run/" + name)

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
    plt.ylim([1e-3, 2.5])
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

    # 4) New results

    folder = "paper_data/new_results/200x3_sr199_v0.07/historyLogs"
    dlra_3layer = pd.read_csv(folder + "/history_001_.csv", delimiter=";")

    plt.plot(dlra_3layer[["acc_val"]], '-')
    plt.plot(dlra_3layer[["acc_train"]], '-.')
    plt.plot(dlra_3layer[["acc_test"]], '--')
    plt.legend(["acc_val", "acc_train", "acc_test"])
    plt.ylim([0.8, 1.05])
    plt.savefig("figures/new_res_007_acc.png", dpi=600)
    plt.clf()

    plt.plot(dlra_3layer[["loss_val"]], '-')
    plt.plot(dlra_3layer[["loss_train"]], '-.')
    plt.plot(dlra_3layer[["loss_test"]], '--')
    plt.legend(["loss_val", "loss_train", "loss_test"])
    plt.ylim([1e-3, 2.5])
    plt.savefig("figures/new_res_007_loss.png", dpi=600)
    plt.yscale('log')
    plt.savefig("figures/new_res_007_loss_log.png", dpi=600)
    plt.clf()
    plt.plot(dlra_3layer[["rank1", "rank2", "rank3"]])
    plt.legend(["layer1", "layer2", "layer3"])
    plt.savefig("figures/new_res_007_ranks.png", dpi=600)
    plt.yscale('log')
    plt.savefig("figures/new_res_007_ranks_log.png", dpi=600)
    plt.clf()

    folder = "paper_data/new_results/200x3_sr199_v0.05/historyLogs"
    dlra_3layer = pd.read_csv(folder + "/history_001_.csv", delimiter=";")

    plt.plot(dlra_3layer[["acc_val"]], '-')
    plt.plot(dlra_3layer[["acc_train"]], '-.')
    plt.plot(dlra_3layer[["acc_test"]], '--')
    plt.legend(["acc_val", "acc_train", "acc_test"])
    plt.ylim([0.8, 1.05])
    plt.savefig("figures/new_res_005_acc.png", dpi=600)
    plt.clf()

    plt.plot(dlra_3layer[["loss_val"]], '-')
    plt.plot(dlra_3layer[["loss_train"]], '-.')
    plt.plot(dlra_3layer[["loss_test"]], '--')
    plt.legend(["loss_val", "loss_train", "loss_test"])
    plt.ylim([1e-3, 2.5])
    plt.savefig("figures/new_res_005_loss.png", dpi=600)
    plt.yscale('log')
    plt.savefig("figures/new_res_005_loss_log.png", dpi=600)
    plt.clf()
    plt.plot(dlra_3layer[["rank1", "rank2", "rank3"]])
    plt.legend(["layer1", "layer2", "layer3"])
    plt.savefig("figures/new_res_005_ranks.png", dpi=600)
    plt.yscale('log')
    plt.savefig("figures/new_res_005_ranks_log.png", dpi=600)
    plt.clf()

    folder = "paper_data/new_results/200x3_sr199_v0.03/historyLogs"
    dlra_3layer = pd.read_csv(folder + "/history_001_.csv", delimiter=";")

    plt.plot(dlra_3layer[["acc_val"]], '-')
    plt.plot(dlra_3layer[["acc_train"]], '-.')
    plt.plot(dlra_3layer[["acc_test"]], '--')
    plt.legend(["acc_val", "acc_train", "acc_test"])
    plt.ylim([0.8, 1.05])
    plt.savefig("figures/new_res_003_acc.png", dpi=600)
    plt.clf()

    plt.plot(dlra_3layer[["loss_val"]], '-')
    plt.plot(dlra_3layer[["loss_train"]], '-.')
    plt.plot(dlra_3layer[["loss_test"]], '--')
    plt.legend(["loss_val", "loss_train", "loss_test"])
    plt.ylim([1e-3, 2.5])
    plt.savefig("figures/new_res_003_loss.png", dpi=600)
    plt.yscale('log')
    plt.savefig("figures/new_res_003_loss_log.png", dpi=600)
    plt.clf()
    plt.plot(dlra_3layer[["rank1", "rank2", "rank3"]])
    plt.legend(["layer1", "layer2", "layer3"])
    plt.savefig("figures/new_res_003_ranks.png", dpi=600)
    plt.yscale('log')
    plt.savefig("figures/new_res_003_ranks_log.png", dpi=600)
    plt.clf()

    folder = "paper_data/new_results/200x3_sr199_v0.01/historyLogs"
    dlra_3layer = pd.read_csv(folder + "/history_001_.csv", delimiter=";")

    plt.plot(dlra_3layer[["acc_val"]], '-')
    plt.plot(dlra_3layer[["acc_train"]], '-.')
    plt.plot(dlra_3layer[["acc_test"]], '--')
    plt.legend(["acc_val", "acc_train", "acc_test"])
    plt.ylim([0.8, 1.05])
    plt.savefig("figures/new_res_001_acc.png", dpi=600)
    plt.clf()

    plt.plot(dlra_3layer[["loss_val"]], '-')
    plt.plot(dlra_3layer[["loss_train"]], '-.')
    plt.plot(dlra_3layer[["loss_test"]], '--')
    plt.legend(["loss_val", "loss_train", "loss_test"])
    plt.ylim([1e-3, 2.5])
    plt.savefig("figures/new_res_001_loss.png", dpi=600)
    plt.yscale('log')
    plt.savefig("figures/new_res_001_loss_log.png", dpi=600)
    plt.clf()
    plt.plot(dlra_3layer[["rank1", "rank2", "rank3"]])
    plt.legend(["layer1", "layer2", "layer3"])
    plt.savefig("figures/new_res_001_ranks.png", dpi=600)
    plt.yscale('log')
    plt.savefig("figures/new_res_001_ranks_log.png", dpi=600)
    plt.clf()

    folder = "paper_data/new_results/200x3_sr199_v0.005/historyLogs"
    dlra_3layer = pd.read_csv(folder + "/history_001_.csv", delimiter=";")

    plt.plot(dlra_3layer[["acc_val"]], '-')
    plt.plot(dlra_3layer[["acc_train"]], '-.')
    plt.plot(dlra_3layer[["acc_test"]], '--')
    plt.legend(["acc_val", "acc_train", "acc_test"])
    plt.ylim([0.8, 1.05])
    plt.savefig("figures/new_res_0005_acc.png", dpi=600)
    plt.clf()

    plt.plot(dlra_3layer[["loss_val"]], '-')
    plt.plot(dlra_3layer[["loss_train"]], '-.')
    plt.plot(dlra_3layer[["loss_test"]], '--')
    plt.legend(["loss_val", "loss_train", "loss_test"])
    plt.ylim([1e-3, 2.5])
    plt.savefig("figures/new_res_0005_loss.png", dpi=600)
    plt.yscale('log')
    plt.savefig("figures/new_res_0005_loss_log.png", dpi=600)
    plt.clf()
    plt.plot(dlra_3layer[["rank1", "rank2", "rank3"]])
    plt.legend(["layer1", "layer2", "layer3"])
    plt.savefig("figures/new_res_0005_ranks.png", dpi=600)
    plt.yscale('log')
    plt.savefig("figures/new_res_0005_ranks_log.png", dpi=600)
    plt.clf()
    """
    return 0


def plot_run(load_folder, save_name):
    plt.clf()
    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 0.7
    markersize = 2.5
    markerwidth = 0.5
    folder = "paper_data/" + load_folder + "/historyLogs"
    dlra_3layer = pd.read_csv(folder + "/history_001_.csv", delimiter=";")

    plt.plot(dlra_3layer[["acc_val"]], '-k')
    plt.plot(dlra_3layer[["acc_train"]], '-.r')
    plt.plot(dlra_3layer[["acc_test"]], '--g')
    plt.legend(["acc_val", "acc_train", "acc_test"])
    plt.ylim([0.8, 1.05])
    plt.savefig("figures/" + save_name + "_acc.png", dpi=500)
    plt.clf()

    plt.plot(dlra_3layer[["loss_val"]], '-k')
    plt.plot(dlra_3layer[["loss_train"]], '-.r')
    plt.plot(dlra_3layer[["loss_test"]], '--g')
    plt.legend(["loss_val", "loss_train", "loss_test"])
    plt.ylim([1e-3, 2.5])
    plt.savefig("figures/" + save_name + "_loss.png", dpi=500)
    plt.yscale('log')
    plt.savefig("figures/" + save_name + "_loss_log.png", dpi=500)
    plt.clf()
    plt.plot(dlra_3layer[["rank1"]], '-k')
    plt.plot(dlra_3layer[["rank2"]], '--r')
    plt.plot(dlra_3layer[["rank3"]], '-.g')
    plt.xlim([0, 1200])
    plt.legend(["rank layer 1", "rank layer 2", "rank layer 3"])
    plt.savefig("figures/" + save_name + "_ranks.png", dpi=500)
    plt.yscale('log')
    plt.savefig("figures/" + save_name + "_ranks_log.png", dpi=500)
    plt.clf()
    return 0


def plot_run4layer(load_folder, save_name):
    plt.clf()
    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 0.7
    markersize = 2.5
    markerwidth = 0.5
    folder = "paper_data/" + load_folder + "/historyLogs"
    dlra_3layer = pd.read_csv(folder + "/history_final.csv", delimiter=";", index_col=None)

    plt.plot(dlra_3layer[["acc_val"]], '-k')
    plt.plot(dlra_3layer[["acc_train"]], '-.r')
    plt.plot(dlra_3layer[["acc_test"]], '--g')
    plt.legend(["acc_val", "acc_train", "acc_test"])
    plt.ylim([0.8, 1.05])
    plt.ylabel("acc")
    plt.xlabel("epoch")

    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 0.5
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.savefig("figures/" + save_name + "_acc.png", dpi=500)
    plt.clf()

    plt.plot(dlra_3layer[["loss_val"]], '-k')
    plt.plot(dlra_3layer[["loss_train"]], '-.r')
    plt.plot(dlra_3layer[["loss_test"]], '--g')
    plt.legend(["loss_val", "loss_train", "loss_test"])
    plt.ylim([1e-3, 2.5])
    plt.ylabel("loss")
    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 0.5
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.savefig("figures/" + save_name + "_loss.png", dpi=500)
    plt.yscale('log')
    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 0.5
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.savefig("figures/" + save_name + "_loss_log.png", dpi=500)
    plt.clf()
    plt.plot(dlra_3layer[["rank1"]], '-k')
    plt.plot(dlra_3layer[["rank2"]], '-r')
    plt.plot(dlra_3layer[["rank3"]], '-g')
    plt.plot(dlra_3layer[["rank4"]], '-b')
    # plt.ylim([10, 120])
    plt.xlim([0, 250])
    plt.xlabel("epoch")
    plt.ylabel("rank")
    plt.legend(["rank layer 1", "rank layer 2", "rank layer 3", "rank layer 4"])
    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 0.5
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.savefig("figures/" + save_name + "_ranks.png", dpi=500)
    plt.yscale('log')
    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 0.5
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.savefig("figures/" + save_name + "_ranks_log.png", dpi=500)
    plt.clf()
    return 0


def plot_timing_exec(load_folder, save_name):
    plt.clf()
    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 0.7
    markersize = 2.5
    markerwidth = 0.5
    folder = "paper_data/" + load_folder
    df = pd.read_csv(folder, delimiter=",")

    plt.plot(df[["rank"]].to_numpy()[1:-1], df[["avg_timing"]].to_numpy()[1:-1], '-ok')
    plt.axhline(y=df[["avg_timing"]].to_numpy()[0], color='r')
    plt.xlim((df[["rank"]].to_numpy()[1], df[["rank"]].to_numpy()[-2]))
    # plt.plot(df[["rank"]].to_numpy()[0], df[["avg_timing"]].to_numpy()[0], 'or')
    plt.xlabel("rank")
    plt.ylabel("time [s]")

    plt.legend(["low-rank", "dense reference"])
    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 0.5
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.savefig("figures/" + save_name + ".png", dpi=500)
    plt.yscale("log")
    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 0.5
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.savefig("figures/" + save_name + "log.png", dpi=500)
    plt.clf()

    return 0


def plot_timing_train(load_folder, save_name):
    plt.clf()
    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 0.7
    markersize = 2.5
    markerwidth = 0.5
    folder = "paper_data/" + load_folder
    df = pd.read_csv(folder, delimiter=",")

    plt.plot(df[["rank"]].to_numpy()[1:-3], df[["avg_timing"]].to_numpy()[1:-3], '-ok')
    plt.axhline(y=df[["avg_timing"]].to_numpy()[0], color='r')
    # plt.plot(df[["rank"]].to_numpy()[0], df[["avg_timing"]].to_numpy()[0], 'or')
    plt.xlabel("rank")
    plt.ylabel("time [s]")
    plt.xlim((df[["rank"]].to_numpy()[1], df[["rank"]].to_numpy()[-4]))

    plt.legend(["low-rank", "dense reference"])
    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 0.5
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.savefig("figures/" + save_name + ".png", dpi=500)
    plt.yscale("log")
    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 0.5
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.savefig("figures/" + save_name + "log.png", dpi=500)
    plt.clf()

    return 0


def print_param_counts():
    # 1) compute parameter counts for low rank layers

    # a) 200 neurons run
    e003_ranks = np.asarray([71, 113, 114, 115])
    e005_ranks = np.asarray([47, 69, 74, 81])
    e007_ranks = np.asarray([37, 40, 45, 48])
    e009_ranks = np.asarray([28, 30, 32, 30])
    e011_ranks = np.asarray([24, 23, 25, 24])
    e013_ranks = np.asarray([17, 16, 22, 19])
    e015_ranks = np.asarray([16, 17, 18, 17])
    e017_ranks = np.asarray([11, 15, 15, 16])

    input_dims = np.asarray([784, 200, 200, 200])
    output_dims = np.asarray([200, 200, 200, 200])

    resdense = 784 * 200 + 3 * 200 * 200 + 200 * 10
    res003 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e003_ranks)
    res005 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e005_ranks)
    res007 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e007_ranks)
    res009 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e009_ranks)
    res011 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e011_ranks)
    res013 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e013_ranks)
    res015 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e015_ranks)
    res017 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e017_ranks)
    print(res003)
    print(res005)
    print(res007)
    print(res009)
    print(res011)
    print(res013)
    print(res015)
    print(res017)
    params_200 = np.asarray(
        [resdense, res003[0], res005[0], res007[0], res009[0], res011[0], res013[0], res015[0], res017[0]])
    accs_200 = np.asarray(
        [98.43, 97.42, 97.17, 97.15, 96.81, 96.76, 96.32, 96.30, 95.59])
    tols = np.asarray(
        [0, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17])
    # plot_acc_over_params(accs_200, params_200, name="accuracy_over_parameter")
    # plot_acc_over_tolerance(accs_200, tols, name="accuracy_over_tolerance")

    # compute ratios
    ratio003 = (float(res003[0]) / float(resdense), float(res003[1]) / float(resdense))
    ratio005 = (float(res005[0]) / float(resdense), float(res005[1]) / float(resdense))
    ratio007 = (float(res007[0]) / float(resdense), float(res007[1]) / float(resdense))
    ratio009 = (float(res009[0]) / float(resdense), float(res009[1]) / float(resdense))
    ratio011 = (float(res011[0]) / float(resdense), float(res011[1]) / float(resdense))
    ratio013 = (float(res013[0]) / float(resdense), float(res013[1]) / float(resdense))
    ratio015 = (float(res015[0]) / float(resdense), float(res015[1]) / float(resdense))
    ratio017 = (float(res017[0]) / float(resdense), float(res017[1]) / float(resdense))
    print(ratio003[1] * 100 - 100)
    print(ratio005[1] * 100 - 100)
    print(ratio007[1] * 100 - 100)
    print(ratio009[1] * 100 - 100)
    print(ratio011[1] * 100 - 100)
    print(ratio013[1] * 100 - 100)
    print(ratio015[1] * 100 - 100)
    print(ratio017[1] * 100 - 100)
    ratios200 = np.asarray([(1, 1), ratio003, ratio005, ratio007, ratio009, ratio011, ratio013, ratio015, ratio017])
    ratios200 = (np.ones((9, 2)) - ratios200) * 100

    # b) 500 neurons run
    e003_ranks = np.asarray([176, 170, 171, 174])
    e005_ranks = np.asarray([81, 104, 111, 117])
    e007_ranks = np.asarray([52, 67, 73, 72])
    e009_ranks = np.asarray([35, 53, 51, 46])
    e011_ranks = np.asarray([27, 40, 37, 38])
    e013_ranks = np.asarray([20, 31, 32, 30])
    e015_ranks = np.asarray([17, 25, 26, 24])
    e017_ranks = np.asarray([13, 21, 24, 20])

    input_dims = np.asarray([784, 500, 500, 500])
    output_dims = np.asarray([500, 500, 500, 500])

    resdense = 784 * 500 + 3 * 500 * 500 + 500 * 10

    res003 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e003_ranks)
    res005 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e005_ranks)
    res007 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e007_ranks)
    res009 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e009_ranks)
    res011 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e011_ranks)
    res013 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e013_ranks)
    res015 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e015_ranks)
    res017 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e017_ranks)
    print(res003)
    print(res005)
    print(res007)
    print(res009)
    print(res011)
    print(res013)
    print(res015)
    print(res017)
    params_500 = np.asarray(
        [resdense, res003[0], res005[0], res007[0], res009[0], res011[0], res013[0], res015[0], res017[0]])
    accs_500 = np.asarray(
        [98.45, 98.47, 98.56, 98.47, 98.31, 98.23, 97.88, 97.70, 97.08])
    tols = np.asarray(
        [0, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17])

    # compute ratios
    ratio003 = (float(res003[0]) / float(resdense), float(res003[1]) / float(resdense))
    ratio005 = (float(res005[0]) / float(resdense), float(res005[1]) / float(resdense))
    ratio007 = (float(res007[0]) / float(resdense), float(res007[1]) / float(resdense))
    ratio009 = (float(res009[0]) / float(resdense), float(res009[1]) / float(resdense))
    ratio011 = (float(res011[0]) / float(resdense), float(res011[1]) / float(resdense))
    ratio013 = (float(res013[0]) / float(resdense), float(res013[1]) / float(resdense))
    ratio015 = (float(res015[0]) / float(resdense), float(res015[1]) / float(resdense))
    ratio017 = (float(res017[0]) / float(resdense), float(res017[1]) / float(resdense))
    print(ratio003[1] * 100 - 100)
    print(ratio005[1] * 100 - 100)
    print(ratio007[1] * 100 - 100)
    print(ratio009[1] * 100 - 100)
    print(ratio011[1] * 100 - 100)
    print(ratio013[1] * 100 - 100)
    print(ratio015[1] * 100 - 100)
    print(ratio017[1] * 100 - 100)

    ratios500 = np.asarray([(1, 1), ratio003, ratio005, ratio007, ratio009, ratio011, ratio013, ratio015, ratio017])
    ratios500 = (np.ones((9, 2)) - ratios500) * 100

    # b) 784 neurons run
    e003_ranks = np.asarray([190, 190, 190, 190])
    e005_ranks = np.asarray([124, 120, 125, 126])
    e007_ranks = np.asarray([76, 86, 85, 83])
    e009_ranks = np.asarray([56, 67, 63, 59])
    e011_ranks = np.asarray([35, 49, 47, 43])
    e013_ranks = np.asarray([29, 35, 38, 34])
    e015_ranks = np.asarray([22, 29, 27, 27])
    e017_ranks = np.asarray([17, 23, 22, 23])

    input_dims = np.asarray([784, 784, 784, 784])
    output_dims = np.asarray([784, 784, 784, 784])

    resdense = 784 * 784 + 3 * 784 * 784 + 784 * 10

    res003 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e003_ranks)
    res005 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e005_ranks)
    res007 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e007_ranks)
    res009 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e009_ranks)
    res011 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e011_ranks)
    res013 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e013_ranks)
    res015 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e015_ranks)
    res017 = compute_param_count(input_dims=input_dims, output_dims=output_dims, layer_ranks=e017_ranks)
    print(res003)
    print(res005)
    print(res007)
    print(res009)
    print(res011)
    print(res013)
    print(res015)
    print(res017)
    params_784 = np.asarray(
        [resdense, res003[0], res005[0], res007[0], res009[0], res011[0], res013[0], res015[0], res017[0]])
    accs_784 = np.asarray(
        [98.63, 98.69, 98.61, 98.58, 98.54, 98.27, 98.14, 97.92, 97.36])
    tols = np.asarray(
        [0, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17])
    # plot_acc_over_params(accs_784, params_784, name="accuracy_over_parameter")
    # plot_acc_over_tolerance(accs_784, tols, name="accuracy_over_tolerance")

    # compute ratios
    ratio003 = (float(res003[0]) / float(resdense), float(res003[1]) / float(resdense))
    ratio005 = (float(res005[0]) / float(resdense), float(res005[1]) / float(resdense))
    ratio007 = (float(res007[0]) / float(resdense), float(res007[1]) / float(resdense))
    ratio009 = (float(res009[0]) / float(resdense), float(res009[1]) / float(resdense))
    ratio011 = (float(res011[0]) / float(resdense), float(res011[1]) / float(resdense))
    ratio013 = (float(res013[0]) / float(resdense), float(res013[1]) / float(resdense))
    ratio015 = (float(res015[0]) / float(resdense), float(res015[1]) / float(resdense))
    ratio017 = (float(res017[0]) / float(resdense), float(res017[1]) / float(resdense))
    print(ratio003[1] * 100 - 100)
    print(ratio005[1] * 100 - 100)
    print(ratio007[1] * 100 - 100)
    print(ratio009[1] * 100 - 100)
    print(ratio011[1] * 100 - 100)
    print(ratio013[1] * 100 - 100)
    print(ratio015[1] * 100 - 100)
    print(ratio017[1] * 100 - 100)
    ratios784 = np.asarray([(1, 1), ratio003, ratio005, ratio007, ratio009, ratio011, ratio013, ratio015, ratio017])
    ratios784 = (np.ones((9, 2)) - ratios784) * 100

    plot_acc_over_params(accs_200, accs_500, accs_784, params_200, params_500, params_784,
                         name="accuracy_over_parameter")
    plot_acc_over_tolerance(tols, accs_200, accs_500, accs_784, name="accuracy_over_tolerance")
    plot_acc_over_compression(ratios200, ratios500, ratios784, accs_200, accs_500, accs_784,
                              name="accuracy_over_compression")

    return 0


def plot_acc_over_params(accs1, accs2, accs3, params1, params2, params3, name):
    plt.clf()
    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 0.7
    markersize = 2.5
    markerwidth = 0.5

    plt.plot(params1, accs1, '-ok')
    plt.plot(params2, accs2, '-og')
    plt.plot(params3, accs3, '-ob')
    plt.plot(params1[0], accs1[0], 'or')
    plt.plot(params2[0], accs2[0], 'or')
    plt.plot(params3[0], accs3[0], 'or')

    plt.xlabel("network weights")
    plt.ylabel("test accuracy")
    plt.xscale("log")
    plt.legend(["200 neurons", "500 neurons", "784 neurons"])
    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 0.5
    ax.set_aspect(abs(np.log(x_right - x_left) / (y_high - y_low)) * 0.7 * 1e-1)

    plt.savefig("figures/" + name + ".png", dpi=500)
    plt.clf()
    return 0


def plot_acc_over_tolerance(tols, accs1, accs2, accs3, name):
    plt.clf()
    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 0.7
    markersize = 2.5
    markerwidth = 0.5

    plt.plot(tols, accs1, '-ok')
    plt.plot(tols, accs2, '-og')
    plt.plot(tols, accs3, '-ob')

    plt.plot(tols[0], accs1[0], 'or')
    plt.plot(tols[0], accs2[0], 'or')
    plt.plot(tols[0], accs3[0], 'or')
    plt.xlabel(r"tolerance $\tau$")
    plt.ylabel("test accuracy")
    plt.legend(["200 neurons", "500 neurons", "784 neurons"])
    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 0.5
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.savefig("figures/" + name + ".png", dpi=500)
    plt.clf()
    return 0


def plot_acc_over_compression(ratios200, ratios500, ratios784, accs1, accs2, accs3, name):
    plt.clf()
    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 0.7
    markersize = 2.5
    markerwidth = 0.5

    plt.plot(ratios200[:, 0], accs1, '-ok')
    plt.plot(ratios500[:, 0], accs2, '-og')
    plt.plot(ratios784[:, 0], accs3, '-ob')

    plt.plot(ratios200[0, 0], accs1[0], 'or')
    plt.plot(ratios500[0, 0], accs2[0], 'or')
    plt.plot(ratios784[0, 0], accs3[0], 'or')
    plt.xlabel(r"compression [%]")
    plt.ylabel("test accuracy")
    plt.legend(["200 neurons", "500 neurons", "784 neurons"])
    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 0.5
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    plt.savefig("figures/" + name + ".png", dpi=500)
    plt.clf()
    return 0


def compute_param_count(input_dims, output_dims, layer_ranks):
    # exec params
    e_params = input_dims * layer_ranks + layer_ranks * output_dims
    res_exec = np.sum(e_params) + output_dims[-1] * 10

    # train_params
    t_params = input_dims * layer_ranks * 2 + layer_ranks * layer_ranks * 4 + 2 * layer_ranks * output_dims
    res_train = np.sum(t_params) + output_dims[-1] * 10
    return res_exec, res_train


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

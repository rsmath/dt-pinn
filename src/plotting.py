import matplotlib.pyplot as plt
import math
import os
from operator import itemgetter
import numpy as np
import json
import torch
import matplotlib as mpl


def algebraic_convergence():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/algebraic_convergence/"
    max_y = 0
    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots()
        for i, order in enumerate(orders):
            # get result files for all training set sizes and both discrete_pinn and vanilla_pinn
            discrete_results = []
            for size in sizes:
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{subfolder}/results.json', 'r') as f:
                    temp_f = min(json.load(f)[f'{train_test}_l2_losses'])
                    discrete_results.append(temp_f)
                    max_y = max(max_y, temp_f)

            ax.plot([str(s) for s in sizes], discrete_results, linestyle='dashed', linewidth=2,
                    marker=markers[str(order)], color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        # only one set of vanilla-PINN results are there for unsupervised
        vanilla_results = []
        for size in sizes:
            with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/{subfolder}/results.json', 'r') as f:
                temp_f = min(json.load(f)[f'{train_test}_l2_losses'])
                vanilla_results.append(temp_f)
                max_y = max(max_y, temp_f)

        ax.plot([str(s) for s in sizes], vanilla_results, color='blue', linestyle='dashed', linewidth=2,
                marker='*', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        plt.xlabel(r"$\mathbf{N}$", fontsize=25)
        plt.title(f'Relative error vs ' + r'$\mathbf{N}$', fontsize=26)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
        ax.xaxis.set_ticks(list(range(len(sizes))))
        ax.xaxis.set_ticklabels([str(s) for s in sizes])
        plt.xticks(rotation=25)
        plt.ylabel(r'Relative error', fontsize=25)
        # plt.ylim(top=3.5e-3)
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.close()

def plot_speedup():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/speedup/"
    max_y = 0

    vanilla_times = []
    for size in sizes:
        with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/unsupervised/results.json', 'r') as f:
            data = json.load(f)
            idx = min(range(len(data[f'{train_test}_l2_losses'])), key=data[f'{train_test}_l2_losses'].__getitem__)
            vanilla_times.append(data['epoch_time'][idx])

    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots()
        for marker_i, order in enumerate(orders):
            discrete_times = []
            for i, size in enumerate(sizes):
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    idx = min(range(len(data[f'{train_test}_l2_losses'])), key=data[f'{train_test}_l2_losses'].__getitem__)
                    discrete_times.append(vanilla_times[i] / data['epoch_time'][idx])
                    max_y = max(max_y, vanilla_times[i] / data['epoch_time'][idx])

            ax.plot([str(s) for s in sizes], discrete_times, linestyle='dashed', linewidth=2,
                    marker=markers[str(order)], color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel(r"$\mathbf{N}$", fontsize=25)
        plt.ylabel(r'Speedup', fontsize=25)
        plt.title(f'Speedup vs ' + r'$\mathbf{N}$', fontsize=26)
        plt.yticks(list(range(0, int(max_y)+2, 3)))
        ax.xaxis.set_ticks(list(range(len(sizes))))
        ax.xaxis.set_ticklabels([str(s) for s in sizes])
        plt.xticks(rotation=25)
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        fig.savefig(save_folder+'combined_plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def plot_speedup_order():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/speedup_vs_order/"
    max_y = 0

    vanilla_times = []
    for size in sizes:
        with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/unsupervised/results.json', 'r') as f:
            data = json.load(f)
            idx = min(range(len(data[f'{train_test}_l2_losses'])), key=data[f'{train_test}_l2_losses'].__getitem__)
            vanilla_times.append(data['epoch_time'][idx])

    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots()
        discrete_times = []
        for marker_i, order in enumerate(orders):
            for i, size in enumerate(sizes):
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    idx = min(range(len(data[f'{train_test}_l2_losses'])), key=data[f'{train_test}_l2_losses'].__getitem__)
                    discrete_times.append(vanilla_times[i] / data['epoch_time'][idx])
                    max_y = max(max_y, vanilla_times[i] / data['epoch_time'][idx])

        ax.plot([str(o) for o in orders], discrete_times, linestyle='dashed', linewidth=2, marker='o', color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'DT-PINN (fp64)')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.yticks(list(range(0, int(max_y)+2)))
        plt.ylabel(r'Speedup', fontsize=25)
        plt.xlabel(r"$\mathbf{p}$", fontsize=25)
        plt.title(f'Speedup vs ' + r'$\mathbf{p}$', fontsize=26)
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        ax.xaxis.set_ticks(list(range(len(orders))))
        ax.xaxis.set_ticklabels([str(s) for s in orders])
        fig.savefig(save_folder+'combined_plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def plot_time_vs_size():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/time_vs_size/"
    max_y = 0
    fig, ax = plt.subplots()

    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        for marker_i, order in enumerate(orders):
            discrete_times = []
            for i, size in enumerate(sizes):
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    idx = min(range(len(data[f'{train_test}_l2_losses'])), key=data[f'{train_test}_l2_losses'].__getitem__)
                    discrete_times.append(data['epoch_time'][idx])
                    max_y = max(max_y, discrete_times[-1])

            ax.plot([str(s) for s in sizes], discrete_times, linestyle='dashed', linewidth=2,
                    marker=markers[str(order)], color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        vanilla_times = []
        for size in sizes:
            with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/unsupervised/results.json', 'r') as f:
                data = json.load(f)
                idx = min(range(len(data[f'{train_test}_l2_losses'])), key=data[f'{train_test}_l2_losses'].__getitem__)
                vanilla_times.append(data['epoch_time'][idx])
                max_y = max(max_y, vanilla_times[-1])

        ax.plot([str(s) for s in sizes], vanilla_times, color='blue', linestyle='dashed', linewidth=2,
                marker='*', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.ylabel(r'Time ($\mathbf{s}$)', fontsize=25)
        plt.xlabel(r"$\mathbf{N}$", fontsize=25)
        # plt.yticks(list(range(0, int(max_y)+20, 200)))
        plt.ticklabel_format(axis='y', style='sci', scilimits=(2,2))
        plt.title(f'Time vs ' + r'$\mathbf{N}$', fontsize=26)
        ax.xaxis.set_ticks(list(range(len(sizes))))
        ax.xaxis.set_ticklabels([str(s) for s in sizes])
        plt.ylim(top=4.3e2)
        plt.xticks(rotation=25)
        plt.legend(frameon=False, loc='upper left', prop=legend_param)
        fig.savefig(save_folder+'combined_plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def plot_l2_vs_order():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/l2_vs_order/"
    max_y = 0
    fig, ax = plt.subplots()

    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        discrete_times = []
        for marker_i, order in enumerate(orders):
            for i, size in enumerate(sizes):
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    min_l2 = min(data[f'{train_test}_l2_losses'])
                    discrete_times.append(min_l2)
                    max_y = max(max_y, discrete_times[-1])

        ax.plot([str(o) for o in orders], discrete_times, linestyle='dashed', linewidth=2, marker='o', color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'DT-PINN (fp64)')

        vanilla_times = []
        for order in orders:
            for size in sizes:
                with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/unsupervised/results.json', 'r') as f:
                    data = json.load(f)
                    min_l2 = min(data[f'{train_test}_l2_losses'])
                    vanilla_times.append(min_l2)
                    max_y = max(max_y, vanilla_times[-1])

        ax.plot([str(order) for order in orders], vanilla_times, color='blue', linestyle='dashed', linewidth=2, marker='*', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.ylabel(r'Relative error', fontsize=25)
        plt.xlabel(r"$\mathbf{p}$", fontsize=25)
        plt.title(f'Relative error vs ' + r'$\mathbf{p}$', fontsize=26)
        ax.xaxis.set_ticks(list(range(len(orders))))
        ax.xaxis.set_ticklabels([str(s) for s in orders])
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        fig.savefig(save_folder+'combined_plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def plot_time_vs_order():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/time_vs_order/"
    max_y = 0
    fig, ax = plt.subplots()

    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        discrete_times = []
        for marker_i, order in enumerate(orders):
            for i, size in enumerate(sizes):
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    idx = min(range(len(data[f'{train_test}_l2_losses'])), key=data[f'{train_test}_l2_losses'].__getitem__)
                    discrete_times.append(data['epoch_time'][idx])
                    max_y = max(max_y, discrete_times[-1])

        ax.plot([str(o) for o in orders], discrete_times, linestyle='dashed', linewidth=2, marker='o', color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'DT-PINN (fp64)')

        vanilla_times = []
        for order in orders:
            for size in sizes:
                with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/unsupervised/results.json', 'r') as f:
                    data = json.load(f)
                    idx = min(range(len(data[f'{train_test}_l2_losses'])), key=data[f'{train_test}_l2_losses'].__getitem__)
                    vanilla_times.append(data['epoch_time'][idx])
                    max_y = max(max_y, vanilla_times[-1])

        ax.plot([str(order) for order in orders], vanilla_times, color='blue', linestyle='dashed', linewidth=2, marker='*', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.ylabel(r'Time ($\mathbf{s}$)', fontsize=25)
        plt.xlabel(r"$\mathbf{p}$", fontsize=25)
        plt.title(f'Time vs ' + r'$\mathbf{p}$', fontsize=26)
        plt.ylim(top=5e3)
        ax.xaxis.set_ticks(list(range(len(orders))))
        ax.xaxis.set_ticklabels([str(s) for s in orders])
        plt.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        fig.savefig(save_folder+'combined_plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def plot_mse():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/mse_vs_epoch/"
    max_epoch = 0
    min_epoch = 1000
    max_y = 0

    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots()
        orders = [2, 3, 4, 5]
        for i, order in enumerate(orders):
            discrete_epochs = []
            for size in sizes:
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    idx = min(range(len(data[f'{train_test}_mse_losses'])), key=data[f'{train_test}_mse_losses'].__getitem__)
                    discrete_epochs.append(data['epochs_list'][idx])
                    max_epoch = max(max_epoch, data['epochs_list'][idx])
                    min_epoch = min(min_epoch, data['epochs_list'][idx])

            ax.plot([str(s) for s in sizes], discrete_epochs, linestyle='dashed', linewidth=2,
                    marker=markers[str(order)], color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        vanilla_epochs = []
        for size in sizes:
            with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/{subfolder}/results.json', 'r') as f:
                data = json.load(f)
                idx = min(range(len(data[f'{train_test}_mse_losses'])), key=data[f'{train_test}_mse_losses'].__getitem__)
                vanilla_epochs.append(data['epochs_list'][idx])
                max_epoch = max(max_epoch, data['epochs_list'][idx])
                min_epoch = min(min_epoch, data['epochs_list'][idx])

        ax.plot([str(s) for s in sizes], vanilla_epochs, color='blue', linestyle='dashed', linewidth=2,
                marker='*', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        # fp64 vanilla-PINN
        vanilla_epochs = []
        for size in sizes:
            with open(f'../gpu_float64_synced_vanilla_results_noprint/{second_pinn}/2/{size}/{subfolder}/results.json', 'r') as f:
                data = json.load(f)
                idx = min(range(len(data[f'{train_test}_mse_losses'])), key=data[f'{train_test}_mse_losses'].__getitem__)
                vanilla_epochs.append(data['epochs_list'][idx])
                max_epoch = max(max_epoch, data['epochs_list'][idx])
                min_epoch = min(min_epoch, data['epochs_list'][idx])

        ax.plot([str(s) for s in sizes], vanilla_epochs, color='orange', linestyle='dashed', linewidth=2,
                marker='v', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN (fp64)')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.ylabel(r"Epochs", fontsize=25)
        plt.ylim(top=1000)
        plt.yticks(list(range(50, 670, 150)))
        plt.title(r'Training epochs vs $\mathbf{N}$', fontsize=26)
        ax.xaxis.set_ticks(list(range(len(sizes))))
        ax.xaxis.set_ticklabels([str(s) for s in sizes])
        plt.xticks(rotation=25)
        plt.xlabel(r'$\mathbf{N}$', fontsize=25)
        plt.legend(frameon=False, loc='upper left', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def plot_epochs():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/epochs/"
    max_epoch = 0
    min_epoch = 1000
    max_y = 0

    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots()
        orders = [5]
        for i, order in enumerate(orders):
            # get result files for all training set sizes and both discrete_pinn and vanilla_pinn
            discrete_results = []
            discrete_epochs = []
            for size in sizes:
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    discrete_results.append(min(data[f'{train_test}_l2_losses']))
                    idx = min(range(len(data[f'{train_test}_l2_losses'])), key=data[f'{train_test}_l2_losses'].__getitem__)
                    discrete_epochs.append(data['epochs_list'][idx])
                    max_y = max(max_y, min(data[f'{train_test}_l2_losses']))
                    max_epoch = max(max_epoch, data['epochs_list'][idx])
                    min_epoch = min(min_epoch, data['epochs_list'][idx])

            ax.plot(discrete_epochs, discrete_results, linestyle='dashed', linewidth=2,
                    marker=markers[str(order)], color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')
            for i, label in enumerate(sizes):
                ax.annotate(str(label), (discrete_epochs[i], discrete_results[i]), fontweight="bold")

        vanilla_results = []
        vanilla_epochs = []
        for size in sizes:
            with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/{subfolder}/results.json', 'r') as f:
                data = json.load(f)
                vanilla_results.append(min(data[f'{train_test}_l2_losses']))
                idx = min(range(len(data[f'{train_test}_l2_losses'])), key=data[f'{train_test}_l2_losses'].__getitem__)
                vanilla_epochs.append(data['epochs_list'][idx])
                max_y = max(max_y, min(data[f'{train_test}_l2_losses']))
                max_epoch = max(max_epoch, data['epochs_list'][idx])
                min_epoch = min(min_epoch, data['epochs_list'][idx])

        ax.plot(vanilla_epochs, vanilla_results, color='blue', linestyle='dashed', linewidth=2,
                marker='*', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')
        for i, label in enumerate(sizes):
            ax.annotate(str(label), (vanilla_epochs[i], vanilla_results[i]), fontweight="bold")

        # fp64 vanilla-PINN
        vanilla_results = []
        vanilla_epochs = []
        for size in sizes:
            with open(f'../gpu_float64_synced_vanilla_results_noprint/{second_pinn}/2/{size}/{subfolder}/results.json', 'r') as f:
                data = json.load(f)
                vanilla_results.append(min(data[f'{train_test}_l2_losses']))
                idx = min(range(len(data[f'{train_test}_l2_losses'])), key=data[f'{train_test}_l2_losses'].__getitem__)
                vanilla_epochs.append(data['epochs_list'][idx])
                max_y = max(max_y, min(data[f'{train_test}_l2_losses']))
                max_epoch = max(max_epoch, data['epochs_list'][idx])
                min_epoch = min(min_epoch, data['epochs_list'][idx])

        ax.plot(vanilla_epochs, vanilla_results, color='orange', linestyle='dashed', linewidth=2,
                marker='v', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN (fp64)')
        for i, label in enumerate(sizes):
            ax.annotate(str(label), (vanilla_epochs[i], vanilla_results[i]), fontweight="bold")

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel(r"Epoch", fontsize=25)
        plt.ylim(0, 0.00018)
        plt.xticks(range(min_epoch, max_epoch, int(max_epoch / 10.)))
        plt.title(f'Relative error', fontsize=26)
        plt.ylabel(r'Relative error', fontsize=25)
        plt.legend(frameon=False, loc='upper left', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def plot_training_loss_epoch():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/{train_test}_loss/"
    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots(dpi=100)
        orders = [2, 3, 4, 5]
        for i, order in enumerate(orders):
            # get result files for all training set sizes and both discrete_pinn and vanilla_pinn
            discrete_results = []
            for size in sizes:
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    discrete_results = list(data[f'{train_test}_losses'][epoch_plot_start::plot_limit])

                ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), discrete_results, linestyle='dashed', linewidth=2, marker=markers[str(order)], color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        vanilla_results = []
        for size in sizes:
            with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/{subfolder}/results.json', 'r') as f:
                data = json.load(f)
                vanilla_results = list(data[f'{train_test}_losses'][epoch_plot_start::plot_limit])

                ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), vanilla_results, color='blue', linestyle='dashed', linewidth=2, marker='*', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel(r"$\mathbf{n}$", fontsize=25)
        plt.title(f'Training loss vs ' + r'$\mathbf{n}$', fontsize=26)
        plt.ylabel(r'Loss', fontsize=25)
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def plot_l2_loss_epoch():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/{train_test}_l2_loss/"
    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots(dpi=100)
        orders = [2, 3, 4, 5]
        for i, order in enumerate(orders):
            # get result files for all training set sizes and both discrete_pinn and vanilla_pinn
            discrete_results = []
            for size in sizes:
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    discrete_results = list(data[f'{train_test}_l2_losses'][epoch_plot_start::plot_limit])

                ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), discrete_results, linestyle='dashed', linewidth=2, marker=markers[str(order)], color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        vanilla_results = []
        for size in sizes:
            with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/{subfolder}/results.json', 'r') as f:
                data = json.load(f)
                vanilla_results = list(data[f'{train_test}_l2_losses'][epoch_plot_start::plot_limit])

                ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), vanilla_results, color='blue', linestyle='dashed', linewidth=2, marker='*', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel(r"Epochs", fontsize=25)
        plt.title(f'Relative error vs ' + r'$\mathbf{n}$', fontsize=26)
        plt.ylabel(r'Relative error', fontsize=25)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def plot_mse_epoch():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/{train_test}_mse_loss/"
    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots(dpi=100)
        orders = [2, 3, 4, 5]
        for i, order in enumerate(orders):
            # get result files for all training set sizes and both discrete_pinn and vanilla_pinn
            discrete_results = []
            for size in sizes:
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    discrete_results = list(data[f'{train_test}_mse_losses'][epoch_plot_start::plot_limit])

                ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), discrete_results, linestyle='dashed', linewidth=2, marker=markers[str(order)], color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        vanilla_results = []
        for size in sizes:
            with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/{subfolder}/results.json', 'r') as f:
                data = json.load(f)
                vanilla_results = list(data[f'{train_test}_mse_losses'][epoch_plot_start::plot_limit])

                ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), vanilla_results, color='blue', linestyle='dashed', linewidth=2, marker='*', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel(r"Epochs", fontsize=25)
        plt.title(f'MSE vs ' + r'$\mathbf{n}$', fontsize=26)
        plt.ylabel(r'Loss', fontsize=25)
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def plot_mse_ampl():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/{train_test}_mse_ampl/"
    max_y = 0
    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots(dpi=100)
        for marker_i, order in enumerate(orders):
            discrete_results = []
            for _, amplitude in enumerate(amplitudes):
                for size in sizes:
                    with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{amplitude}/{subfolder}/results.json', 'r') as f:
                        data = json.load(f)
                        discrete_results.append(min(data[f'{train_test}_mse_losses']))
                        max_y = max(max_y, discrete_results[-1])

            ax.plot([str(s) for s in amplitudes], discrete_results, linestyle='dashed', linewidth=2, marker=markers[str(order)], color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        vanilla_results = []
        for amplitude in amplitudes:
            for size in sizes:
                with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/{amplitude}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    vanilla_results.append(min(data[f'{train_test}_mse_losses']))
                    max_y = max(max_y, vanilla_results[-1])


        ax.plot([str(s) for s in amplitudes], vanilla_results, color='blue', linestyle='dashed', linewidth=2, marker='*', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel(r"$\mathbf{\alpha}$", fontsize=25)
        plt.title(f'MSE vs ' + r'$\mathbf{\alpha}$', fontsize=26)
        ax.xaxis.set_ticks(list(range(len(amplitudes))))
        ax.xaxis.set_ticklabels([str(s) for s in amplitudes])
        plt.xticks(rotation=25)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
        plt.ylabel(r'Loss', fontsize=25)
        plt.legend(frameon=False, loc='upper left', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def plot_time_ampl():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/{train_test}_time_ampl/"
    max_y = 0
    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots(dpi=100)
        for marker_i, order in enumerate(orders):
            discrete_results = []
            for _, amplitude in enumerate(amplitudes):
                for size in sizes:
                    with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{amplitude}/{subfolder}/results.json', 'r') as f:
                        data = json.load(f)
                        idx = min(range(len(data[f'{train_test}_mse_losses'])), key=data[f'{train_test}_mse_losses'].__getitem__)
                        discrete_results.append(data[f'epoch_time'][idx])
                        max_y = max(max_y, discrete_results[-1])

            ax.plot([str(s) for s in amplitudes], discrete_results, linestyle='dashed', linewidth=2, marker=markers[str(order)], color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        vanilla_results = []
        for amplitude in amplitudes:
            for size in sizes:
                with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/{amplitude}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    idx = min(range(len(data[f'{train_test}_mse_losses'])), key=data[f'{train_test}_mse_losses'].__getitem__)
                    vanilla_results.append(data['epoch_time'][idx])
                    max_y = max(max_y, vanilla_results[-1])


        ax.plot([str(s) for s in amplitudes], vanilla_results, color='blue', linestyle='dashed', linewidth=2, marker='*', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel(r"$\mathbf{\alpha}$", fontsize=25)
        plt.title(f'Time vs ' + r'$\mathbf{\alpha}$', fontsize=26)
        ax.xaxis.set_ticks(list(range(len(amplitudes))))
        ax.xaxis.set_ticklabels([str(s) for s in amplitudes])
        plt.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        plt.xticks(rotation=25)
        plt.ylabel(r'Time ($\mathbf{s}$)', fontsize=25)
        plt.legend(frameon=False, loc='upper left', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def plot_speedup_ampl():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/{train_test}_speedup_ampl/"
    max_y = 0

    vanilla_results = []
    for amplitude in amplitudes:
        for size in sizes:
            with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/{amplitude}/unsupervised/results.json', 'r') as f:
                data = json.load(f)
                idx = min(range(len(data[f'{train_test}_mse_losses'])), key=data[f'{train_test}_mse_losses'].__getitem__)
                vanilla_results.append(data['epoch_time'][idx])
                max_y = max(max_y, vanilla_results[-1])

    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots(dpi=100)

        for marker_i, order in enumerate(orders):
            discrete_results = []
            for amp_i, amplitude in enumerate(amplitudes):
                for size in sizes:
                    with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{amplitude}/{subfolder}/results.json', 'r') as f:
                        data = json.load(f)
                        idx = min(range(len(data[f'{train_test}_mse_losses'])), key=data[f'{train_test}_mse_losses'].__getitem__)
                        discrete_results.append(vanilla_results[amp_i] / data['epoch_time'][idx])
                        max_y = max(max_y, discrete_results[-1])

            ax.plot([str(s) for s in amplitudes], discrete_results, linestyle='dashed', linewidth=2, marker=markers[str(order)], color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel(r"$\mathbf{\alpha}$", fontsize=25)
        plt.title(f'Speedup vs ' + r'$\mathbf{\alpha}$', fontsize=26)
        ax.xaxis.set_ticks(list(range(len(amplitudes))))
        ax.xaxis.set_ticklabels([str(s) for s in amplitudes])
        plt.ylabel(r'Speedup', fontsize=25)
        plt.xticks(rotation=25)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(2,2))
        plt.legend(frameon=False, loc='upper left', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def plot_epochs_separate():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    folder = f"../{global_save_folder}/epochs/"
    max_epoch = 0

    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        for order in orders:
            fig, ax = plt.subplots()
            # get result files for all training set sizes and both discrete_pinn and vanilla_pinn
            discrete_results = []
            discrete_epochs = []
            for size in sizes:
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    discrete_results.append(min(data[f'{train_test}_l2_losses']))
                    idx = min(range(len(data[f'{train_test}_l2_losses'])), key=data[f'{train_test}_l2_losses'].__getitem__)
                    discrete_epochs.append(data['epochs_list'][idx])
                    max_epoch = max(max_epoch, data['epochs_list'][idx])

            ax.plot(discrete_epochs, discrete_results, linestyle='dashed', linewidth=2, color='red',
                    marker='s', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')
            for i, label in enumerate(sizes):
                ax.annotate(str(label), (discrete_epochs[i], discrete_results[i]), fontweight="bold")

            vanilla_results = []
            vanilla_epochs = []
            for size in sizes:
                with open(f'../{vanilla_results_folder}/{second_pinn}/2/{size}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    vanilla_results.append(min(data[f'{train_test}_l2_losses']))
                    idx = min(range(len(data[f'{train_test}_l2_losses'])), key=data[f'{train_test}_l2_losses'].__getitem__)
                    vanilla_epochs.append(data['epochs_list'][idx])
                    max_epoch = max(max_epoch, data['epochs_list'][idx])

            ax.plot(vanilla_epochs, vanilla_results, color='blue', linestyle='dashed', linewidth=2,
                    marker='*', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')
            for i, label in enumerate(sizes):
                ax.annotate(str(label), (vanilla_epochs[i], vanilla_results[i]), fontweight="bold")

            save_folder = folder + f"{order}/"
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)

            plt.xlabel('Epochs', fontsize=25)
            plt.xticks(range(0, max_epoch, int(max_epoch / 10.)), fontsize=12)
            plt.title(rf'Epochs taken to reach lowest relative {train_test} set error', fontsize=26)
            plt.ylabel(r'$\|\| {\bf L} \|\|_2$ error', fontsize=25)
            plt.legend(frameon=False, loc='upper right', prop=legend_param)
            fig.savefig(save_folder+'plot.png', bbox_inches='tight')
            plt.cla()
            plt.clf()
            plt.close()

def autograd_pde_residual():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/autograd_pde_residual/"

    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots()
        for i, order in enumerate(orders):
            # get result files for all training set sizes and both discrete_pinn and vanilla_pinn
            with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{largest_size}/{subfolder}/results.json', 'r') as f:
                data = json.load(f)
                discrete_results = list(data[f'{train_test}_pde_residual'])[epoch_plot_start::plot_limit]
                ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), discrete_results, linestyle='dashed', linewidth=2, marker=markers[str(order)], color=marker_colors[str(order)], markevery=50, ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        with open(f'../{vanilla_results_folder}/{second_pinn}/2/{largest_size}/{subfolder}/results.json', 'r') as f:
            data = json.load(f)
            vanilla_results = list(data[f'{train_test}_pde_residual'][epoch_plot_start::plot_limit])
            ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), vanilla_results, color='blue', linestyle='dashed', linewidth=2, marker='*', markevery=50, ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel('Epochs', fontsize=25)
        # plt.ylim(0.01, 0.06)
        plt.yticks(fontsize=12)
        plt.ylabel(r'$\Delta u - f$', fontsize=25)
        plt.title(f'{train_test} Poisson autograd interior residual', fontsize=26)
        plt.xticks(fontsize=12)
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def autograd_boundary_residual():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/autograd_boundary_residual/"

    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots()
        for i, order in enumerate(orders):
            # get result files for all training set sizes and both discrete_pinn and vanilla_pinn
            with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{largest_size}/{subfolder}/results.json', 'r') as f:
                data = json.load(f)
                discrete_results = list(data[f'{train_test}_boundary_residual'])[epoch_plot_start::plot_limit]
                ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), discrete_results, linestyle='dashed', linewidth=2, marker=markers[str(order)], color=marker_colors[str(order)], markevery=50, ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        with open(f'../{vanilla_results_folder}/{second_pinn}/2/{largest_size}/{subfolder}/results.json', 'r') as f:
            data = json.load(f)
            vanilla_results = list(data[f'{train_test}_boundary_residual'][epoch_plot_start::plot_limit])
            ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), vanilla_results, color='blue', linestyle='dashed', linewidth=2, marker='*', markevery=50, ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel('Epochs', fontsize=25)
        plt.yticks(fontsize=12)
        plt.ylabel(r'$(\alpha) ( {\bf n} \cdot \nabla u ) + \beta u - g$', fontsize=25)
        plt.title(f'{train_test} Poisson autograd boundary residual', fontsize=26)
        plt.xticks(fontsize=12)
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def discrete_pde_residual():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/discrete_pde_residual/"

    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots()
        for i, order in enumerate(orders):
            # get result files for all training set sizes and both discrete_pinn and vanilla_pinn
            with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{largest_size}/{subfolder}/results.json', 'r') as f:
                data = json.load(f)
                discrete_results = list(data[f'training_discrete_pde_residual'])[epoch_plot_start::plot_limit]
                ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), discrete_results, linestyle='dashed', linewidth=2, marker=markers[str(order)], color=marker_colors[str(order)], markevery=50, ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        with open(f'../{vanilla_results_folder}/{second_pinn}/2/{largest_size}/{subfolder}/results.json', 'r') as f:
            data = json.load(f)
            vanilla_results = list(data[f'training_pde_residual'][epoch_plot_start::plot_limit])
            ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), vanilla_results, color='blue', linestyle='dashed', linewidth=2, marker='*', markevery=50, ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel('Epochs', fontsize=25)
        plt.yticks(fontsize=12)
        plt.ylabel(r'${\bf L} u - f$', fontsize=25)
        plt.title(f'training Poisson discrete interior residual', fontsize=26)
        plt.xticks(fontsize=12)
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def discrete_boundary_residual():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/discrete_boundary_residual/"

    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots()
        for i, order in enumerate(orders):
            # get result files for all training set sizes and both discrete_pinn and vanilla_pinn
            with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{largest_size}/{subfolder}/results.json', 'r') as f:
                data = json.load(f)
                discrete_results = list(data[f'training_discrete_boundary_residual'])[epoch_plot_start::plot_limit]
                ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), discrete_results, linestyle='dashed', linewidth=2, marker=markers[str(order)], color=marker_colors[str(order)], markevery=50, ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        with open(f'../{vanilla_results_folder}/{second_pinn}/2/{largest_size}/{subfolder}/results.json', 'r') as f:
            data = json.load(f)
            vanilla_results = list(data[f'training_boundary_residual'][epoch_plot_start::plot_limit])
            ax.plot(list(data['epochs_list'][epoch_plot_start::plot_limit]), vanilla_results, color='blue', linestyle='dashed', linewidth=2, marker='*', markevery=50, ms=14, markeredgewidth=1, markeredgecolor='black', label=f'vanilla-PINN ({vanilla_precision})')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel('Epochs', fontsize=25)
        plt.yticks(fontsize=12)
        plt.ylabel(r'${\bf B} u - g$', fontsize=25)
        plt.title(f'training Poisson discrete boundary residual', fontsize=26)
        plt.xticks(fontsize=12)
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def effect_of_depth_time():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/{train_test}_time/"
    max_y = 0
    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots(dpi=100)
        for marker_i, order in enumerate(orders):
            discrete_results = []
            for _, layer in enumerate(layers):
                for size in sizes:
                    with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{layer}/{subfolder}/results.json', 'r') as f:
                        data = json.load(f)
                        discrete_results.append(data[f'discrete_time'][0])
                        max_y = max(max_y, discrete_results[-1])

            ax.plot([str(s) for s in layers], discrete_results, linestyle='dashed', linewidth=2, marker=markers[str(order)], color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        vanilla_results = []
        fp64_vanilla_results = []
        for layer in layers:
            for size in sizes:
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{layer}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    vanilla_results.append(data['fp32_autograd_time'][0])
                    max_y = max(max_y, vanilla_results[-1])
                    fp64_vanilla_results.append(data['fp64_autograd_time'][0])
                    max_y = max(max_y, fp64_vanilla_results[-1])


        ax.plot([str(s) for s in layers], vanilla_results, color='blue', linestyle='dashed', linewidth=2, marker='*', ms=14, markeredgewidth=1, markeredgecolor='black', label='autograd (fp32)')
        ax.plot([str(s) for s in layers], fp64_vanilla_results, color='orange', linestyle='dashed', linewidth=2, marker='v', ms=14, markeredgewidth=1, markeredgecolor='black', label='autograd (fp64)')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel(r"$\mathbf{s}$", fontsize=25)
        plt.title(f'Time vs ' + r'$\mathbf{s}$', fontsize=26)
        ax.xaxis.set_ticks(list(range(len(layers))))
        ax.xaxis.set_ticklabels([str(s) for s in layers])
        plt.ylabel(r'Time ($\mathbf{s}$)', fontsize=25)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-1,-1))
        plt.ylim(top=0.155)
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

def effect_of_depth_l2():
    plt.rcParams["figure.figsize"] = figure_figsize
    plt.rcParams['ytick.labelsize'] = ytick_size
    plt.rcParams['xtick.labelsize'] = xtick_size
    save_folder = f"../{global_save_folder}/{train_test}_l2/"
    max_y = 0
    for supervised in [False]:
        subfolder = 'supervised' if supervised else 'unsupervised'
        fig, ax = plt.subplots(dpi=100)
        for marker_i, order in enumerate(orders):
            discrete_results = []
            for _, layer in enumerate(layers):
                for size in sizes:
                    with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{layer}/{subfolder}/results.json', 'r') as f:
                        data = json.load(f)
                        discrete_results.append(data[f'discrete_fp64_autograd_l2'][0])
                        max_y = max(max_y, discrete_results[-1])

            ax.semilogy([str(s) for s in layers], discrete_results, linestyle='dashed', linewidth=2, marker=markers[str(order)], color=marker_colors[str(order)], ms=14, markeredgewidth=1, markeredgecolor='black', label=f'{order_string[str(order)]}')

        vanilla_results = []
        for layer in layers:
            for size in sizes:
                with open(f'../{discrete_results_folder}/{first_pinn}/{order}/{size}/{layer}/{subfolder}/results.json', 'r') as f:
                    data = json.load(f)
                    vanilla_results.append(data['fp32_autograd_fp64_autograd_l2'][0])
                    max_y = max(max_y, vanilla_results[-1])

        ax.semilogy([str(s) for s in layers], discrete_results, linestyle='dashed', linewidth=2, marker='*', color='blue', ms=14, markeredgewidth=1, markeredgecolor='black', label=f'autograd ({vanilla_precision})')

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel(r"$\mathbf{s}$", fontsize=25)
        plt.title(f'Relative error vs ' + r'$\mathbf{s}$', fontsize=26)
        ax.xaxis.set_ticks(list(range(len(layers))))
        ax.xaxis.set_ticklabels([str(s) for s in layers])
        plt.ylim(top=1e-2)
        plt.ylabel('Relative error', fontsize=25)
        plt.legend(frameon=False, loc='upper right', prop=legend_param)
        fig.savefig(save_folder+'plot.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

if __name__ == '__main__':
    sizes = [582, 828, 1663, 2236, 3196, 4977, 6114, 8767, 19638] # 2d
    # sizes = [810, 3080, 4325, 6218] # 3d
    # sizes = [828] # heat equation
    # sizes = [828] # noisy equation
    # sizes = [19638] # effect of depth

    orders = [2, 3, 4, 5]
    amplitudes = [0.001, 0.0020, 0.0040, 0.0080, 0.0160, 0.0320, 0.0640, 0.1280, 0.2560]
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    plot_limit = 300
    epoch_plot_start = 60

    order_string = {
            "2": "p = 2 (fp64)",
            "3": "p = 3 (fp64)",
            "4": "p = 4 (fp64)",
            "5": "p = 5 (fp64)"
    }
    marker_colors = {
            "2": "blue",
            "3": "orange",
            "4": "green",
            "5": "red"
    }
    markers = {
            "2": "o",
            "3": "s",
            "4": "D",
            "5": "^"
    }

    PRECISION = torch.float32
    precision_string = "float32" if PRECISION == torch.float32 else "float64"

    network_PRECISIONS = [torch.float32]
    for network_PRECISION in network_PRECISIONS:
        network_precision_string = "float32" if network_PRECISION == torch.float32 else "float64"

        global_save_folder = f""

        discrete_results_folder = f""
        vanilla_results_folder = f""

        vanilla_precision = "fp32"

        first_pinn = "discrete_pinn"
        second_pinn = "vanilla_pinn"

        train_test = "test"

        largest_size = sizes[-1]

        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["legend.fontsize"] = 23
        figure_figsize = (11, 11)
        ytick_size = 23
        xtick_size = 23

        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
        plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'

        legend_param = {'weight': 'normal'}

        # effect_of_depth_time()
        # effect_of_depth_l2()
        # plot_training_loss_epoch()
        # plot_l2_loss_epoch()
        # plot_mse_epoch()
        # plot_time_vs_order()
        # plot_speedup_order()
        # plot_l2_vs_order()
        # plot_mse_ampl()
        # plot_time_ampl()
        # plot_speedup_ampl()
        # algebraic_convergence()
        # plot_speedup()
        # plot_time_vs_size()
        # plot_mse()
        # plot_epochs()
        # plot_time_combined()
        # autograd_pde_residual()
        # autograd_boundary_residual()
        # discrete_pde_residual()
        # discrete_boundary_residual()

import os
import pickle
import matplotlib
import matplotlib.pyplot as plt

import numpy as np


PATH_TO = os.path.join(os.getcwd(), 'info_cifar10/')
SAVE_TO = os.path.join(PATH_TO, 'heat_dir_new/')
if not os.path.exists(SAVE_TO):
    os.mkdir(SAVE_TO)

epochs = [100, 200,400, 800]
widths_str = ['025',  '05', '1', '2']
widths = [0.25, 0.5, 1, 2]
losses = ['ce', 'mse', 'fl', 'ls']
seeds = [0,1,2]

heat_dict = dict()
loss_dict = dict()
var_dict = dict()
for loss in losses:
    if loss == 'ce':
        pass
    else:
        heat_dict[loss] = dict()
        heat_dict[loss]['test'] = np.zeros([len(epochs), len(widths)])

    loss_dict[loss] = dict()
    loss_dict[loss]['test'] = np.zeros([len(epochs), len(widths)])

    var_dict[loss] = dict()
    var_dict[loss]['test'] = np.zeros([len(seeds), len(epochs), len(widths)])

for loss in losses:
    if loss == 'ce':
        pass
    else:
        for i, epoch in enumerate(epochs):
            for j, width in enumerate(widths_str):
                train_last_top1_ce, val_best_top1_ce, best_val_test_top1_ce = 0, 0, 0
                train_last_top1_loss, val_best_top1_loss, best_val_test_top1_loss = 0, 0, 0
                num_s = 0
                for k,s in enumerate(seeds):
                    filename = 'wideres50-{width}-cifar10-{loss}-epochs{epoch}-{seed}/'.format(width=width, loss=loss,
                                                                                                epoch=epoch, seed=s)+'info.pkl'
                    filename_ce = 'wideres50-{width}-cifar10-{loss}-epochs{epoch}-{seed}/'.format(width=width, loss='ce',
                                                                                                epoch=epoch, seed=s)+'info.pkl'
                    path_to_info = os.path.join(PATH_TO, filename)
                    path_to_info_ce = os.path.join(PATH_TO, filename_ce)
                    if os.path.exists(path_to_info) and os.path.exists(path_to_info_ce):
                        num_s += 1
                        with open(path_to_info, 'rb') as f:
                            info = pickle.load(f)
                        with open(path_to_info_ce, 'rb') as f:
                            info_ce = pickle.load(f)

                        best_val_test_top1_ce += info_ce['best_val_test_top1']
                        best_val_test_top1_loss += info['best_val_test_top1']


                        var_dict['ce']['test'][k, len(epochs) - 1 - i, j] = info_ce['best_val_test_top1']
                        var_dict[loss]['test'][k, len(epochs) - 1 - i, j] = info['best_val_test_top1']

                loss_dict['ce']['test'][len(epochs) - 1 - i, j] = best_val_test_top1_ce/num_s
                loss_dict[loss]['test'][len(epochs) - 1 - i, j] = best_val_test_top1_loss/num_s

                heat_dict[loss]['test'][len(epochs) - 1 - i, j] = abs((best_val_test_top1_ce-best_val_test_top1_loss)/num_s)

def plot_heat_map(heat_array):
    for loss in heat_array.keys():
        for mode in heat_array[loss].keys():
            fig, ax = plt.subplots(figsize=(12, 9))
            im = ax.imshow(heat_array[loss][mode])

            # We want to show all ticks...
            ax.set_xticks(np.arange(len(widths)))
            ax.set_yticks(np.arange(len(epochs)))

            plt.xlabel('Width', fontsize=40)
            plt.ylabel('Epochs', fontsize=40)

            # Loop over data dimensions and create text annotations.
            for i in range(len(epochs)):
                for j in range(len(widths)):
                    text = ax.text(j, i, '%.3f'%round(heat_array[loss][mode][i, j], 3),
                                   ha="center", va="center", color="w", fontsize=30)

            # ... and label them with the respective list entries
            ax.set_xticklabels(widths, fontsize=40)

            ax.set_yticklabels(epochs[::-1], fontsize=40)

            ax.set_title("%s "%mode + r'$|Acc_{ce}-Acc_{%s}|$'%(loss), fontsize=40)
            fig.tight_layout()
            plt.colorbar(im)
            fig.savefig(os.path.join(SAVE_TO, "heatmap-{loss}-{mode}.png".format(loss=loss, mode=mode)), bbox_inches='tight')


def plot_loss_map(loss_array):
    for loss in loss_array.keys():
        for mode in loss_array[loss].keys():
            fig, ax = plt.subplots(figsize=(12, 9))
            im = ax.imshow(loss_array[loss][mode])

            # We want to show all ticks...
            ax.set_xticks(np.arange(len(widths)))
            ax.set_yticks(np.arange(len(epochs)))

            plt.xlabel('Width', fontsize=40)
            plt.ylabel('Epochs', fontsize=40)

            # Loop over data dimensions and create text annotations.
            for i in range(len(epochs)):
                for j in range(len(widths)):
                    text = ax.text(j, i, '%.3f'%round(loss_array[loss][mode][i, j], 3),
                                   ha="center", va="center", color="w", fontsize=30)

            # ... and label them with the respective list entries
            ax.set_xticklabels(widths, fontsize=40)

            ax.set_yticklabels(epochs[::-1], fontsize=40)

            ax.set_title("%s "%mode + r'$Acc_{%s}$'%(loss), fontsize=40)
            fig.tight_layout()
            plt.colorbar(im)
            fig.savefig(os.path.join(SAVE_TO, "lossmap-{loss}-{mode}.png".format(loss=loss, mode=mode)), bbox_inches='tight')

def plot_test_line(loss_array ,epochs):
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.grid(True)
    # plt.show()

    legends = list()
    colors = ['c', 'b', 'g', 'r']
    log_epochs = np.log2(np.array(epochs)//100)
    for loss in losses:
        for w in range(len(widths)):
            test_acc1 = loss_array[loss]['test'][::-1, w]
            # test_acc1 = test_acc1.tolist().reverse()
            linestyle = 'solid' if loss == 'ce' else 'dashed'
            plt.plot(log_epochs, test_acc1, colors[w], linewidth=5, linestyle=linestyle, alpha=0.7)

            legends.append( '{loss}-width-{width}'.format(loss=loss, width=widths[w]))

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Test Acc1', fontsize=40)
    plt.xticks(log_epochs, fontsize=30)
    plt.yticks(np.arange(70, 80.1, 2), fontsize=30)
    ax.set_xticklabels(epochs, fontsize=40)

    plt.legend(legends, fontsize=20)
    plt.axis([log_epochs[0], log_epochs[-1], 70, 80])
    fig.savefig(SAVE_TO + "test1_line.png", bbox_inches='tight')

def plot_var_map(var_dict):
    ce = var_dict['ce']['test'].reshape((1, 3, -1))
    mse = var_dict['mse']['test'].reshape((1, 3, -1))
    ls = var_dict['ls']['test'].reshape((1, 3, -1))
    fl = var_dict['fl']['test'].reshape((1, 3, -1))
    var = np.concatenate([ce, mse, ls, fl], axis=0)
    var = var.reshape((12, -1))

    var = np.var(var, axis=0).reshape((4,4))

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(var)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(widths)))
    ax.set_yticks(np.arange(len(epochs)))

    plt.xlabel('Width', fontsize=40)
    plt.ylabel('Epochs', fontsize=40)


    # Loop over data dimensions and create text annotations.
    for i in range(len(epochs)):
        for j in range(len(widths)):
            text = ax.text(j, i, '%.3f' % round(var[i, j], 3),
                           ha="center", va="center", color="w", fontsize=30)

    # ... and label them with the respective list entries
    ax.set_xticklabels(widths, fontsize=40)

    ax.set_yticklabels(epochs[::-1], fontsize=40)

    ax.set_title('Variance', fontsize=40)
    fig.tight_layout()
    plt.colorbar(im)
    fig.savefig(os.path.join(SAVE_TO, "variance.png"), bbox_inches='tight')


if __name__ == "__main__":
    plot_heat_map(heat_dict)
    plot_loss_map(loss_dict)
    plot_var_map(var_dict)


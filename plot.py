import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

PATH_TO_INFO = os.path.join(os.getcwd(), 'info_res50_cifar100/')

PATH_TO_INFO_ce_1 = os.path.join(PATH_TO_INFO, 'wideres50-2-cifar100-ce-epochs800-1/' + 'info.pkl')

PATH_TO_INFO_fl_1 = os.path.join(PATH_TO_INFO, 'wideres50-2-cifar100-fl-epochs800-1/' + 'info.pkl')

PATH_TO_INFO_ls_1 = os.path.join(PATH_TO_INFO, 'wideres50-2-cifar100-ls-epochs800-1/' + 'info.pkl')

PATH_TO_INFO_mse_1 = os.path.join(PATH_TO_INFO, 'wideres50-2-cifar100-mse-epochs800-1/' + 'info_epochs.pkl')
#
out_path = os.path.join(os.path.dirname(PATH_TO_INFO), 'imgs_epochs/')
if not os.path.exists(out_path):
    os.makedirs(out_path, exist_ok=True)
#
with open(PATH_TO_INFO_ce_1, 'rb') as f:
    info_ETFfc_ce_1 = pickle.load(f)
with open(PATH_TO_INFO_fl_1, 'rb') as f:
    info_ETFfc_fl_1 = pickle.load(f)
with open(PATH_TO_INFO_ls_1, 'rb') as f:
    info_ETFfc_ls_1 = pickle.load(f)
with open(PATH_TO_INFO_mse_1, 'rb') as f:
    info_ETFfc_mse_1 = pickle.load(f)

x_max = 800
XTICKS = np.array([0, 25, 50, 75, 100]) * (x_max//100)
xrange = np.arange(1, 101, 1) * (x_max//100)
legend = ['CE', 'LS', 'FL', 'MSE']


def plot_collapse():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    print(xrange.shape)
    print(np.array(info_ETFfc_ce_1['collapse_metric']).shape)

    plt.plot(xrange, np.array(info_ETFfc_ce_1['collapse_metric']), 'b', marker='v', ms=16, markevery=200, linewidth=5,
             alpha=0.7)
    plt.plot(xrange, np.array(info_ETFfc_ls_1['collapse_metric']), 'g', marker='s', ms=16, markevery=200, linewidth=5,
             alpha=0.7)
    plt.plot(xrange, np.array(info_ETFfc_fl_1['collapse_metric']), 'r', marker='X', ms=16, markevery=200, linewidth=5,
             alpha=0.7)
    plt.plot(xrange, np.array(info_ETFfc_mse_1['collapse_metric']), 'c', marker='o', ms=16, markevery=200, linewidth=5,
             alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_1$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 12.1, 4), fontsize=30)

    plt.legend(legend, fontsize=30)
    plt.axis([0, x_max, -0.4, 12])

    fig.savefig(out_path + "NC1.png", bbox_inches='tight')


def plot_ETF():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, info_ETFfc_ce_1['ETF_metric'], 'b', marker='v', ms=16, markevery=200, linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_ls_1['ETF_metric'], 'g', marker='s', ms=16, markevery=200, linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_fl_1['ETF_metric'], 'r', marker='X', ms=16, markevery=200, linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_mse_1['ETF_metric'], 'c', marker='o', ms=16, markevery=200, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_2$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(-0.2, 1.21, .2), fontsize=30)

    plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE', 'COS'], fontsize=30)
    plt.legend(legend, fontsize=30)

    plt.axis([0, x_max, -0.02, 1.2])

    fig.savefig(out_path + "NC2.png", bbox_inches='tight')


def plot_WH_relation():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, info_ETFfc_ce_1['WH_relation_metric'], 'b', marker='v', ms=16, markevery=200, linewidth=5,
             alpha=0.7)
    plt.plot(xrange, info_ETFfc_ls_1['WH_relation_metric'], 'g', marker='s', ms=16, markevery=200, linewidth=5,
             alpha=0.7)
    plt.plot(xrange, info_ETFfc_fl_1['WH_relation_metric'], 'r', marker='X', ms=16, markevery=200, linewidth=5,
             alpha=0.7)
    plt.plot(xrange, info_ETFfc_mse_1['WH_relation_metric'], 'c', marker='o', ms=16, markevery=200, linewidth=5,
             alpha=0.7)

    #

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_3$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 1.21, 0.2), fontsize=30)

    plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE', 'COS'], fontsize=30)
    plt.legend(legend, fontsize=30)

    plt.axis([0, x_max, 0, 1.2])

    fig.savefig(out_path + "NC3.png", bbox_inches='tight')


def plot_residual():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, info_ETFfc_ce_1['Wh_b_relation_metric'], 'b', marker='v', ms=16, markevery=200, linewidth=5,
             alpha=0.7)
    plt.plot(xrange, info_ETFfc_ls_1['Wh_b_relation_metric'], 'g', marker='s', ms=16, markevery=200, linewidth=5,
             alpha=0.7)
    plt.plot(xrange, info_ETFfc_fl_1['Wh_b_relation_metric'], 'r', marker='X', ms=16, markevery=200, linewidth=5,
             alpha=0.7)
    plt.plot(xrange, info_ETFfc_mse_1['Wh_b_relation_metric'], 'c', marker='o', ms=16, markevery=200, linewidth=5,
             alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_4$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 3.01, 0.5), fontsize=30)

    plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE', 'COS'], fontsize=30)
    plt.legend(legend, fontsize=30)

    plt.axis([0, x_max, 0, 3.0])

    fig.savefig(out_path + "NC4.png", bbox_inches='tight')


def plot_train_acc():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, info_ETFfc_ce_1['train_acc1'], 'b', marker='v', ms=16, markevery=200, linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_ls_1['train_acc1'], 'g', marker='s', ms=16, markevery=200, linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_fl_1['train_acc1'], 'r', marker='X', ms=16, markevery=200, linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_mse_1['train_acc1'], 'c', marker='o', ms=16, markevery=200, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Training accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(40, 110, 20), fontsize=30)
    plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE', 'COS'], fontsize=30)
    plt.legend(legend, fontsize=30)

    plt.axis([0, x_max, 20, 102])

    fig.savefig(out_path + "train-acc.png", bbox_inches='tight')


def plot_val_acc():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, info_ETFfc_ce_1['val_acc1'], 'b', marker='v', ms=16, markevery=200, linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_ls_1['val_acc1'], 'g', marker='s', ms=16, markevery=200, linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_fl_1['val_acc1'], 'r', marker='X', ms=16, markevery=200, linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_mse_1['val_acc1'], 'c', marker='o', ms=16, markevery=200, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Val accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 81, 20), fontsize=30)
    plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE', 'COS'], fontsize=30)
    plt.legend(legend, fontsize=30)

    plt.axis([0, x_max, 0, 80])

    fig.savefig(out_path + "val-acc.png", bbox_inches='tight')


def plot_test_acc():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, info_ETFfc_ce_1['test_acc1'], 'b', marker='v', ms=16, markevery=200, linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_ls_1['test_acc1'], 'g', marker='s', ms=16, markevery=200, linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_fl_1['test_acc1'], 'r', marker='X', ms=16, markevery=200, linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_mse_1['test_acc1'], 'c', marker='o', ms=16, markevery=200, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Testing accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 81, 20), fontsize=30)
    plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE', 'COS'], fontsize=30)

    plt.axis([0, x_max, 0, 80])
    plt.legend(legend, fontsize=30)

    fig.savefig(out_path + "test-acc.png", bbox_inches='tight')


def main():
    plot_collapse()
    plot_ETF()
    plot_WH_relation()
    plot_residual()

    plot_train_acc()
    plot_val_acc()
    plot_test_acc()


if __name__ == "__main__":
    main()
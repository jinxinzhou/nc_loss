import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

PATH_TO_INFO = os.path.join(os.getcwd(), 'model_directory/')

PATH_TO_INFO_ce_1 = os.path.join(PATH_TO_INFO, 'resnet50-cifar100-cross-entropy-epochs700-cos/' + 'info.pkl')
PATH_TO_INFO_ce_2 = os.path.join(PATH_TO_INFO, 'resnet50-cifar100-cross-entropy-epochs700-cos-123/' + 'info.pkl')
PATH_TO_INFO_ce_3 = os.path.join(PATH_TO_INFO, 'resnet50-cifar100-cross-entropy-epochs700-cos-321/' + 'info.pkl')

# PATH_TO_INFO_fl_1 = os.path.join(PATH_TO_INFO, 'wideres26-20-cifar100-focalloss3-epochs350/'+'info.pkl')
# PATH_TO_INFO_fl_2 = os.path.join(PATH_TO_INFO, 'wideres26-20-cifar100-focalloss3-epochs350-123/'+'info.pkl')
# PATH_TO_INFO_fl_3 = os.path.join(PATH_TO_INFO, 'wideres26-20-cifar100-focalloss3-epochs350-321/'+'info.pkl')

# PATH_TO_INFO_ls_1 = os.path.join(PATH_TO_INFO, 'wideres26-20-cifar100-ls0.05-epochs350/'+'info.pkl')
# PATH_TO_INFO_ls_2 = os.path.join(PATH_TO_INFO, 'wideres26-20-cifar100-ls0.05-epochs350-123/'+'info.pkl')
# PATH_TO_INFO_ls_3 = os.path.join(PATH_TO_INFO, 'wideres26-20-cifar100-ls0.05-epochs350-321/'+'info.pkl')

PATH_TO_INFO_mse_1 = os.path.join(PATH_TO_INFO, 'resnet50-cifar100-mse-epochs700-cos/' + 'info.pkl')
PATH_TO_INFO_mse_2 = os.path.join(PATH_TO_INFO, 'resnet50-cifar100-mse-epochs700-cos/' + 'info.pkl')
PATH_TO_INFO_mse_3 = os.path.join(PATH_TO_INFO, 'resnet50-cifar100-mse-epochs700-cos-321/' + 'info.pkl')
# PATH_TO_INFO_cos_loss = os.path.join(PATH_TO_INFO, 'resnet50-cifar100-cos-loss-epochs350/'+'info.pkl')
#
out_path = os.path.join(os.path.dirname(PATH_TO_INFO), 'figures/resnet50-cifar100-epochs700-cos/')
if not os.path.exists(out_path):
    os.makedirs(out_path, exist_ok=True)
#
with open(PATH_TO_INFO_ce_1, 'rb') as f:
    info_ETFfc_ce_1 = pickle.load(f)
with open(PATH_TO_INFO_ce_2, 'rb') as f:
    info_ETFfc_ce_2 = pickle.load(f)
with open(PATH_TO_INFO_ce_3, 'rb') as f:
    info_ETFfc_ce_3 = pickle.load(f)

# with open(PATH_TO_INFO_fl_1, 'rb') as f:
#     info_ETFfc_fl_1 = pickle.load(f)
# with open(PATH_TO_INFO_fl_2, 'rb') as f:
#     info_ETFfc_fl_2 = pickle.load(f)
# with open(PATH_TO_INFO_fl_3, 'rb') as f:
#     info_ETFfc_fl_3 = pickle.load(f)

# with open(PATH_TO_INFO_ls_1, 'rb') as f:
#     info_ETFfc_ls_1 = pickle.load(f)
# with open(PATH_TO_INFO_ls_2, 'rb') as f:
#     info_ETFfc_ls_2 = pickle.load(f)
# with open(PATH_TO_INFO_ls_3, 'rb') as f:
#     info_ETFfc_ls_3 = pickle.load(f)
#
with open(PATH_TO_INFO_mse_1, 'rb') as f:
    info_ETFfc_mse_1 = pickle.load(f)
with open(PATH_TO_INFO_mse_2, 'rb') as f:
    info_ETFfc_mse_2 = pickle.load(f)
with open(PATH_TO_INFO_mse_3, 'rb') as f:
    info_ETFfc_mse_3 = pickle.load(f)

# with open(PATH_TO_INFO_cos_loss, 'rb') as f:
#     info_ETFfc_cos_loss = pickle.load(f)


# XTICKS = [0, 50, 100, 150, 200, 250, 300, 350]
# xrange = np.arange(0,350, 1)
# x_max = 350

XTICKS = [0, 100, 200, 300, 400, 500, 600, 700]
xrange = np.arange(10, 701, 10)
x_max = 700


def plot_collapse():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, (
                np.array(info_ETFfc_ce_1['collapse_metric']) + np.array(info_ETFfc_ce_2['collapse_metric']) + np.array(
            info_ETFfc_ce_3['collapse_metric'])) / 3, 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_ls_1['collapse_metric'])+np.array(info_ETFfc_ls_2['collapse_metric'])+np.array(info_ETFfc_ls_3['collapse_metric']))/3, 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_fl_1['collapse_metric'])+np.array(info_ETFfc_fl_2['collapse_metric'])+np.array(info_ETFfc_fl_3['collapse_metric']))/3, 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(xrange, (np.array(info_ETFfc_mse_1['collapse_metric']) + np.array(
        info_ETFfc_mse_2['collapse_metric']) + np.array(info_ETFfc_mse_3['collapse_metric'])) / 3, 'r', marker='X',
             ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_1$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 12.1, 4), fontsize=30)

    #     plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE'], fontsize=30)
    plt.legend(['CE', 'MSE'], fontsize=30)
    plt.axis([0, x_max, -0.4, 12])

    fig.savefig(out_path + "NC1.png", bbox_inches='tight')


def plot_ETF():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, (np.array(info_ETFfc_ce_1['ETF_metric']) + np.array(info_ETFfc_ce_2['ETF_metric']) + np.array(
        info_ETFfc_ce_3['ETF_metric'])) / 3, 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_ls_1['ETF_metric'])+np.array(info_ETFfc_ls_2['ETF_metric'])+np.array(info_ETFfc_ls_3['ETF_metric']))/3, 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_fl_1['ETF_metric'])+np.array(info_ETFfc_fl_2['ETF_metric'])+np.array(info_ETFfc_fl_3['ETF_metric']))/3, 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(xrange, (np.array(info_ETFfc_mse_1['ETF_metric']) + np.array(info_ETFfc_mse_2['ETF_metric']) + np.array(
        info_ETFfc_mse_3['ETF_metric'])) / 3, 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, info_ETFfc_cos_loss['ETF_metric'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
    #              alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_2$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(-0.2, 1.21, .2), fontsize=30)

    #     plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE', 'COS'], fontsize=30)
    plt.legend(['CE', 'MSE'], fontsize=30)

    plt.axis([0, x_max, -0.02, 1.2])

    fig.savefig(out_path + "NC2.png", bbox_inches='tight')


def plot_WH_relation():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, (np.array(info_ETFfc_ce_1['WH_relation_metric']) + np.array(
        info_ETFfc_ce_2['WH_relation_metric']) + np.array(info_ETFfc_ce_3['WH_relation_metric'])) / 3, 'c', marker='v',
             ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_ls_1['WH_relation_metric'])+np.array(info_ETFfc_ls_2['WH_relation_metric'])+np.array(info_ETFfc_ls_3['WH_relation_metric']))/3, 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_fl_1['WH_relation_metric'])+np.array(info_ETFfc_fl_2['WH_relation_metric'])+np.array(info_ETFfc_fl_3['WH_relation_metric']))/3, 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(xrange, (np.array(info_ETFfc_mse_1['WH_relation_metric']) + np.array(
        info_ETFfc_mse_2['WH_relation_metric']) + np.array(info_ETFfc_mse_3['WH_relation_metric'])) / 3, 'r',
             marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, info_ETFfc_cos_loss['WH_relation_metric'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
    #              alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_3$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 1.21, 0.2), fontsize=30)

    #     plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE', 'COS'], fontsize=30)
    plt.legend(['CE', 'MSE'], fontsize=30)

    plt.axis([0, x_max, 0, 1.2])

    fig.savefig(out_path + "NC3.png", bbox_inches='tight')


def plot_residual():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, (np.array(info_ETFfc_ce_1['Wh_b_relation_metric']) + np.array(
        info_ETFfc_ce_2['Wh_b_relation_metric']) + np.array(info_ETFfc_ce_3['Wh_b_relation_metric'])) / 3, 'c',
             marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_ls_1['Wh_b_relation_metric'])+np.array(info_ETFfc_ls_2['Wh_b_relation_metric'])+np.array(info_ETFfc_ls_3['Wh_b_relation_metric']))/3, 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_fl_1['Wh_b_relation_metric'])+np.array(info_ETFfc_fl_2['Wh_b_relation_metric'])+np.array(info_ETFfc_fl_3['Wh_b_relation_metric']))/3, 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(xrange, (np.array(info_ETFfc_mse_1['Wh_b_relation_metric']) + np.array(
        info_ETFfc_mse_2['Wh_b_relation_metric']) + np.array(info_ETFfc_mse_3['Wh_b_relation_metric'])) / 3, 'r',
             marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, info_ETFfc_cos_loss['Wh_b_relation_metric'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
    #              alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_4$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 5.01, 1), fontsize=30)

    #     plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE', 'COS'], fontsize=30)
    plt.legend(['CE', 'MSE'], fontsize=30)

    plt.axis([0, x_max, 0, 5.0])

    fig.savefig(out_path + "NC4.png", bbox_inches='tight')


def plot_train_acc():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, (np.array(info_ETFfc_ce_1['train_acc1']) + np.array(info_ETFfc_ce_2['train_acc1']) + np.array(
        info_ETFfc_ce_3['train_acc1'])) / 3, 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_ls_1['train_acc1'])+np.array(info_ETFfc_ls_2['train_acc1'])+np.array(info_ETFfc_ls_3['train_acc1']))/3, 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_fl_1['train_acc1'])+np.array(info_ETFfc_fl_2['train_acc1'])+np.array(info_ETFfc_fl_3['train_acc1']))/3, 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(xrange, (np.array(info_ETFfc_mse_1['train_acc1']) + np.array(info_ETFfc_mse_2['train_acc1']) + np.array(
        info_ETFfc_mse_3['train_acc1'])) / 3, 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, info_ETFfc_cos_loss['train_acc1'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
    #              alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Training accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(40, 110, 20), fontsize=30)
    #     plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE', 'COS'], fontsize=30)
    plt.legend(['CE', 'MSE'], fontsize=30)

    plt.axis([0, x_max, 20, 102])

    fig.savefig(out_path + "train-acc.png", bbox_inches='tight')


def plot_val_acc():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, (np.array(info_ETFfc_ce_1['val_acc1']) + np.array(info_ETFfc_ce_2['val_acc1']) + np.array(
        info_ETFfc_ce_3['val_acc1'])) / 3, 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_ls_1['val_acc1'])+np.array(info_ETFfc_ls_2['val_acc1'])+np.array(info_ETFfc_ls_3['val_acc1']))/3, 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_fl_1['val_acc1'])+np.array(info_ETFfc_fl_2['val_acc1'])+np.array(info_ETFfc_fl_3['val_acc1']))/3, 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(xrange, (np.array(info_ETFfc_mse_1['val_acc1']) + np.array(info_ETFfc_mse_2['val_acc1']) + np.array(
        info_ETFfc_mse_3['val_acc1'])) / 3, 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, info_ETFfc_cos_loss['val_acc1'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
    #              alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Val accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(30, 81, 10), fontsize=30)
    #     plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE', 'COS'], fontsize=30)
    plt.legend(['CE', 'MSE'], fontsize=30)

    plt.axis([0, x_max, 30, 80])

    fig.savefig(out_path + "val-acc.png", bbox_inches='tight')


def plot_test_acc():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, (np.array(info_ETFfc_ce_1['test_acc1']) + np.array(info_ETFfc_ce_2['test_acc1']) + np.array(
        info_ETFfc_ce_3['test_acc1'])) / 3, 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_ls_1['test_acc1'])+np.array(info_ETFfc_ls_2['test_acc1'])+np.array(info_ETFfc_ls_3['test_acc1']))/3, 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_fl_1['test_acc1'])+np.array(info_ETFfc_fl_2['test_acc1'])+np.array(info_ETFfc_fl_3['test_acc1']))/3, 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(xrange, (np.array(info_ETFfc_mse_1['test_acc1']) + np.array(info_ETFfc_mse_2['test_acc1']) + np.array(
        info_ETFfc_mse_3['test_acc1'])) / 3, 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, info_ETFfc_cos_loss['test_acc1'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
    #              alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Testing accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(30, 81, 10), fontsize=30)
    #     plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE', 'COS'], fontsize=30)
    plt.legend(['CE', 'MSE'], fontsize=30)

    plt.axis([0, x_max, 30, 80])

    fig.savefig(out_path + "test-acc.png", bbox_inches='tight')


def plot_cos_margin_distribution():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    ITICKS = [0, 10000, 20000, 30000, 40000, 50000]
    # ------------------------------------- plot for figure 6 ----------------------------------------------------------
    plt.plot(xrange, info_ETFfc_cross_entropy['all_cos_margin'][-1], 'c', marker='v', ms=16, markevery=10000,
             linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_focalloss['all_cos_margin'][-1], 'b', marker='o', ms=16, markevery=10000, linewidth=5,
             alpha=0.7)
    plt.plot(xrange, info_ETFfc_labelsmoothing['all_cos_margin'][-1], 'g', marker='s', ms=16, markevery=10000,
             linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_mse['all_cos_margin'][-1], 'r', marker='X', ms=16, markevery=10000, linewidth=5,
             alpha=0.7)
    #     plt.plot(xrange, info_ETFfc_cos_loss['all_cos_margin'][-1], 'purple', marker='>', ms=16, markevery=10000, linewidth=5,
    #              alpha=0.7)

    plt.xlabel('Index', fontsize=40)
    plt.ylabel('Cos Margin Distribution', fontsize=40)
    plt.xticks(ITICKS, fontsize=30)

    plt.yticks(np.arange(-1.2, 1.3, 0.4), fontsize=30)
    plt.legend(['d=5', 'd=8', 'd=9', 'd=10', 'd=512'], fontsize=30)
    plt.axis([0, 50000, -1.2, 1.3])

    fig.savefig(out_path + "c_margin_dist.png", bbox_inches='tight')


def plot_nuclear():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info_ETFfc_cross_entropy['nuclear_metric'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_focalloss['nuclear_metric'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_labelsmoothing['nuclear_metric'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_mse['nuclear_metric'], 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(info_ETFfc_cos_loss['nuclear_metric'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
    #              alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Avg. class NF_metric', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 6.1, 1), fontsize=30)

    plt.legend(['CE', 'FL,' + r'$\gamma=3$', 'LS' + r'$\beta=0.05$', 'MSE', 'COS'], fontsize=30)

    plt.axis([0, x_max, 0, 6])

    fig.savefig(out_path + "nuclear.png", bbox_inches='tight')


def plot_avg_cos_margin():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(xrange, (
                np.array(info_ETFfc_ce_1['avg_cos_margin']) + np.array(info_ETFfc_ce_2['avg_cos_margin']) + np.array(
            info_ETFfc_ce_3['avg_cos_margin'])) / 3, 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_ls_1['avg_cos_margin'])+np.array(info_ETFfc_ls_2['avg_cos_margin'])+np.array(info_ETFfc_ls_3['avg_cos_margin']))/3, 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, (np.array(info_ETFfc_fl_1['avg_cos_margin'])+np.array(info_ETFfc_fl_2['avg_cos_margin'])+np.array(info_ETFfc_fl_3['avg_cos_margin']))/3, 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(xrange, (
                np.array(info_ETFfc_mse_1['avg_cos_margin']) + np.array(info_ETFfc_mse_2['avg_cos_margin']) + np.array(
            info_ETFfc_mse_3['avg_cos_margin'])) / 3, 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)
    #     plt.plot(xrange, info_ETFfc_cos_loss['avg_cos_margin'], 'purple', marker='>', ms=16, markevery=25, linewidth=5,
    #              alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Avg. Cos Margin', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(-0.4, 1.21, 0.4), fontsize=30)

    #     plt.legend(['CE', 'LS,'+r'$\alpha=0.05$', 'FL,'+r'$\beta=3$', 'MSE', 'COS'], fontsize=30)
    plt.legend(['CE', 'MSE'], fontsize=30)

    plt.axis([0, x_max, -0.4, 1.2])

    fig.savefig(out_path + "avg-cmargin.png", bbox_inches='tight')


def plot_part_cos_margin_distribution(k=100):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    ITICKS = np.arange(0, 6 * (k // 5), k // 5)

    plt.plot(xrange, info_ETFfc_cross_entropy['all_cos_margin'][-1][:k], 'c', marker='v', ms=16, markevery=k // 5,
             linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_focalloss['all_cos_margin'][-1][:k], 'b', marker='o', ms=16, markevery=k // 5,
             linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_labelsmoothing['all_cos_margin'][-1][:k], 'g', marker='s', ms=16, markevery=k // 5,
             linewidth=5, alpha=0.7)
    plt.plot(xrange, info_ETFfc_mse['all_cos_margin'][-1][:k], 'r', marker='v', ms=16, markevery=k // 5, linewidth=5,
             alpha=0.7)
    #     plt.plot(xrange, info_ETFfc_cos_loss['all_cos_margin'][-1][:k], 'purple', marker='>', ms=16, markevery=k // 5,
    #              linewidth=5, alpha=0.7)

    plt.xlabel('Index', fontsize=40)
    plt.ylabel(r'$\mathcal{P}_{CM}$', fontsize=40)
    plt.xticks(ITICKS, fontsize=30)

    plt.yticks(np.arange(-1.2, 1.21, 0.4), fontsize=30)
    plt.legend(['CE', 'FL,' + r'$\gamma=3$', 'LS,' + r'$\beta=0.05$', 'MSE', 'COS'], fontsize=30)
    plt.axis([0, k, -1.2, 1.2])

    fig.savefig(out_path + "c_margin_dist_part.png", bbox_inches='tight')


def main():
    plot_collapse()
    plot_ETF()
    plot_WH_relation()
    plot_residual()

    plot_train_acc()
    plot_val_acc()
    plot_test_acc()

    #     plot_cos_margin_distribution()

    #     # plot_nuclear()
    plot_avg_cos_margin()


#     plot_part_cos_margin_distribution(k=1000)

if __name__ == "__main__":
    main()
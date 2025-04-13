from sklearn.manifold import TSNE
from sklearn import datasets
import matplotlib.pyplot as plt
from MIA_SP_v8 import MIA
# from AED_SP_wo_str import AED
import numpy as np
import torch
import re
import pandas as pd


def motivation():
    train_record = pd.read_csv('../dataset/Ciao/train_record.csv')
    activity = train_record.groupby('uid').count().reset_index().rename(columns={'iid': 'count'})
    train_record = train_record.merge(activity, on='uid')
    group1 = train_record[(train_record['count'] <= 10)].reset_index(drop=True)[['uid', 'iid']]
    group2 = train_record[(train_record['count'] > 10) & (train_record['count'] <= 20)].reset_index(drop=True)[
        ['uid', 'iid']]
    group3 = train_record[(train_record['count'] > 20) & (train_record['count'] <= 50)].reset_index(drop=True)[
        ['uid', 'iid']]
    group4 = train_record[(train_record['count'] > 50)].reset_index(drop=True)[['uid', 'iid']]

    group1_u = torch.tensor(list(set(group1['uid'])), dtype=torch.int)
    group2_u = torch.tensor(list(set(group2['uid'])), dtype=torch.int)
    group3_u = torch.tensor(list(set(group3['uid'])), dtype=torch.int)
    group4_u = torch.tensor(list(set(group4['uid'])), dtype=torch.int)
    # group2_u = set(group2['uid'])
    # group3_u = set(group3['uid'])
    # group4_u = set(group4['uid'])

    return group1_u, group2_u, group3_u, group4_u


def t_SNE_v2(model: MIA):
    u1_tensor, u2_tensor, u3_tensor, u4_tensor = motivation()

    users_preference, items_preference = model.pGcn()
    users_structure, items_structure = model.structure()

    # users_embed = users_preference
    users_embed = torch.cat((users_preference, users_structure), dim=-1)
    # items_embed = torch.cat((items_preference, items_structure), dim=-1)
    x = torch.cat((users_embed[u1_tensor], users_embed[u2_tensor], users_embed[u3_tensor], users_embed[u4_tensor]),
                  dim=0)

    # x = torch.cat((users_preference[u1_tensor], users_preference[u2_tensor], users_preference[u3_tensor],
    #                users_preference[u4_tensor]), dim=0)
    # x = torch.cat((users_structure[u1_tensor], users_structure[u2_tensor], users_structure[u3_tensor],
    #                users_structure[u4_tensor]), dim=0)
    y = torch.cat((torch.tensor([0], dtype=torch.int).repeat(u1_tensor.shape[0]),
                   torch.tensor([1], dtype=torch.int).repeat(u2_tensor.shape[0]),
                   torch.tensor([2], dtype=torch.int).repeat(u3_tensor.shape[0]),
                   torch.tensor([3], dtype=torch.int).repeat(u4_tensor.shape[0])), dim=0)

    x = x.detach().numpy()
    y = y.detach().numpy()
    # 创建一个TSNE实例，设置参数
    t_sne = TSNE(n_components=2, init='pca', random_state=0)

    # 加载Iris数据集 鸢尾花
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target

    x_2d = t_sne.fit_transform(x)

    # 可视化结果
    target_names = ['group1', 'group2', 'group3', 'group4']
    target_ids = range(len(target_names))

    plt.figure(figsize=(4, 3), dpi=100)
    colors = 'r', 'g', 'b', 'o'

    for target_id, color, target_name in zip(target_ids, colors, target_names):
        plt.scatter(x_2d[y == target_id, 0], x_2d[y == target_id, 1], s=0.5)
        # plt.legend(fontsize='large', handlelength=10, handleheight=10)
        # plt.legend(loc='upper right', title_fontsize=10, fontsize=10)
        # plt.scatter(x_2d[y == target_id, 0], x_2d[y == target_id, 1], c=color, label=target_name, s=1)
        # plt.scatter(x_2d[y == target_id, 0], x_2d[y == target_id, 1], c=color, s=1)

    plt.legend(labels=target_names, loc='upper right', fontsize=10, markerscale=5)
    # plt.title('item embedding(initialization)', fontsize=11)
    # plt.title('item embedding(convergence)', fontsize=11)
    plt.show()


# GCN特征和structure特征可视化聚类对比
def t_SNE(model: MIA):
    user_preference, item_preference = model.pGcn()
    users_structure, items_structure = model.structure()
    x = torch.cat((item_preference, items_structure), dim=0)
    y = torch.cat((torch.zeros(size=(item_preference.shape[0],)), torch.ones(size=(items_structure.shape[0],))), dim=0)

    x = x.detach().numpy()
    y = y.detach().numpy()
    # 创建一个TSNE实例，设置参数
    t_sne = TSNE(n_components=2, init='pca', random_state=0)

    # 加载Iris数据集 鸢尾花
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target

    x_2d = t_sne.fit_transform(x)

    # 可视化结果
    target_names = ['e-PRE', 'e-STR']
    target_ids = range(len(target_names))

    plt.figure(figsize=(4, 3), dpi=100)
    colors = 'r', 'g'

    for target_id, color, target_name in zip(target_ids, colors, target_names):
        plt.scatter(x_2d[y == target_id, 0], x_2d[y == target_id, 1], s=0.5)
        # plt.legend(fontsize='large', handlelength=10, handleheight=10)
        # plt.legend(loc='upper right', title_fontsize=10, fontsize=10)
        # plt.scatter(x_2d[y == target_id, 0], x_2d[y == target_id, 1], c=color, label=target_name, s=1)
        # plt.scatter(x_2d[y == target_id, 0], x_2d[y == target_id, 1], c=color, s=1)

    plt.legend(labels=target_names, loc='upper right', fontsize=10, markerscale=5)
    plt.title('item embedding(initialization)', fontsize=11)
    # plt.title('item embedding(convergence)', fontsize=11)
    plt.show()


def extract_metrics():
    log_path = r'D:\project\AED\output\lastfm_AED_2024-04-13-13-38\log\train.log.DESKTOP-OO50KSK.admin.log.INFO.20240413-133828.18708'
    metric_result = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for i in range(50, 1020, 51):
            line = lines[i]
            start = line.find('Test')
            line = line[start:]
            result = re.findall(r"\d+\.\d{5}", line)

            recall = float(result[1])
            metric_result.append(recall)

    x = list(range(50, 1020, 51))

    plt.plot(x, metric_result, "--", label="pool num=5")

    # # 设置字体
    font1 = {'family': 'SimSun', 'weight': 'normal', 'size': 14}
    plt.rc('font', **font1)
    plt.rcParams["axes.unicode_minus"] = False

    # x轴刻度标签设置
    plt.xticks(x, fontproperties=font1)
    # y轴标签数值范围设置
    # 标题设置
    plt.title("pool num 参数影响", fontproperties=font1)
    plt.xlabel("epochs", fontproperties=font1)
    plt.ylabel("Recall@20", fontproperties=font1)
    plt.xticks(range(50, 1020, 204))
    # 图例设置
    plt.legend()
    plt.show()


# structure Model消融
def structure_ablation():
    # 设置字体, 解决中文乱码问题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 解决图像中的'-'负号的乱码问题
    plt.rcParams['axes.unicode_minus'] = False

    x_labels = ['Ciao', 'Yahoo!R3', 'coat']
    y_aed_wo_str = [0.0810 * 100, 0.1664 * 100, 0.1642 * 100]
    y_aed = [0.0849 * 100, 0.1711 * 100, 0.1700 * 100]
    legend_labels = ['AED', 'AED-w/o str']
    y = [y_aed, y_aed_wo_str]
    fig = plt.figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(facecolor='white')
    # 红橙黄绿青蓝紫
    # color_list = ['#FF0000', '#FF8C00', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#800080']
    x_loc = np.arange(3)
    # x轴上每个刻度上能容纳的柱子的总的宽度设为0.8
    total_width = 0.8
    # 由y值可以看出x轴每个刻度上一共有3组数据, 也即3个柱子
    total_num = 3
    # 每个柱子的宽度用each_width表示
    each_width = total_width / total_num
    if total_num % 2 == 0:
        x1 = x_loc - (total_num / 2 - 1) * each_width - each_width / 2
    else:
        x1 = x_loc - ((total_num - 1) / 2) * each_width
    x_list = [x1 + each_width * i for i in range(total_num)]
    print(x_list)
    # 这里颜色设置成 橙色:"#FF8C00"
    for i in range(0, len(y)):
        ax.bar(x_list[i], y[i], width=each_width, label=legend_labels[i])
    ax.set_xticks(x_loc)
    ax.set_xticklabels(x_labels)
    # ax.grid(True, ls=':', color='b', alpha=0.3)
    ax.set_ylabel('Recall@20(%)', fontweight='bold')
    # 添加双轴
    fig.legend(loc='upper left', ncol=5, handlelength=0.9,
               handleheight=0.9, fontsize='small')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.tick_params(labelsize=8)
    plt.show()


# 点击干涉消融
def interference_ablation():
    # 设置字体, 解决中文乱码问题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 解决图像中的'-'负号的乱码问题
    plt.rcParams['axes.unicode_minus'] = False

    x_labels = ['Ciao', 'Yahoo!R3', 'coat']
    y_aed_w_pre_inter = [0.0766 * 100, 0.1659 * 100, 0.1665 * 100]
    y_aed_wo_str_inter = [0.0778 * 100, 0.1606 * 100, 0.1453 * 100]
    y_aed = [0.0849 * 100, 0.1711 * 100, 0.1700 * 100]
    legend_labels = ['AED', 'w/o STR-I', 'w/ GCN-I']
    y = [y_aed, y_aed_wo_str_inter, y_aed_w_pre_inter]
    fig = plt.figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(facecolor='white')
    # 红橙黄绿青蓝紫
    # color_list = ['#FF0000', '#FF8C00', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#800080']
    x_loc = np.arange(3)
    # x轴上每个刻度上能容纳的柱子的总的宽度设为0.8
    total_width = 0.8
    # 由y值可以看出x轴每个刻度上一共有3组数据, 也即3个柱子
    total_num = 3
    # 每个柱子的宽度用each_width表示
    each_width = total_width / total_num
    if total_num % 2 == 0:
        x1 = x_loc - (total_num / 2 - 1) * each_width - each_width / 2
    else:
        x1 = x_loc - ((total_num - 1) / 2) * each_width
    x_list = [x1 + each_width * i for i in range(total_num)]
    print(x_list)
    # 这里颜色设置成 橙色:"#FF8C00"
    for i in range(0, len(y)):
        ax.bar(x_list[i], y[i], width=each_width)
    ax.set_xticks(x_loc)
    ax.set_xticklabels(x_labels)
    # ax.grid(True, ls=':', color='b', alpha=0.3)
    ax.set_ylabel('Recall@20(%)', fontweight='bold')
    # 添加双轴
    fig.legend(legend_labels, loc='upper left', bbox_to_anchor=(0.18, 0.95), fontsize=8, markerscale=5)
    fig.tight_layout()
    # fig.subplots_adjust(top=0.9)
    # plt.tick_params(labelsize=8)
    plt.show()


# 融合负采样消融
def fusion_negative_ablation():
    # 设置字体, 解决中文乱码问题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 解决图像中的'-'负号的乱码问题
    plt.rcParams['axes.unicode_minus'] = False

    x_labels = ['Ciao', 'Yahoo!R3', 'coat']
    y_aed_w_pre_inter = [0.0778 * 100, 0.1606 * 100, 0.1453 * 100]
    y_aed_wo_str_inter = [0.0778 * 100, 0.1606 * 100, 0.1453 * 100]
    y_aed_wo_str = [0.0810 * 100, 0.1664 * 100, 0.1642 * 100]
    y_aed = [0.0844 * 100, 0.1679 * 100, 0.1700 * 100]
    legend_labels = ['AED', 'AED-w/o str(i)', 'AED-w/o str-i', 'AED-w pre(i)']
    y = [y_aed, y_aed_wo_str, y_aed_wo_str_inter]
    fig = plt.figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(facecolor='white')
    # 红橙黄绿青蓝紫
    # color_list = ['#FF0000', '#FF8C00', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#800080']
    x_loc = np.arange(3)
    # x轴上每个刻度上能容纳的柱子的总的宽度设为0.8
    total_width = 0.8
    # 由y值可以看出x轴每个刻度上一共有3组数据, 也即3个柱子
    total_num = 3
    # 每个柱子的宽度用each_width表示
    each_width = total_width / total_num
    if total_num % 2 == 0:
        x1 = x_loc - (total_num / 2 - 1) * each_width - each_width / 2
    else:
        x1 = x_loc - ((total_num - 1) / 2) * each_width
    x_list = [x1 + each_width * i for i in range(total_num)]
    print(x_list)
    # 这里颜色设置成 橙色:"#FF8C00"
    for i in range(0, len(y)):
        ax.bar(x_list[i], y[i], width=each_width, label=legend_labels[i])
    ax.set_xticks(x_loc)
    ax.set_xticklabels(x_labels)
    # ax.grid(True, ls=':', color='b', alpha=0.3)
    ax.set_ylabel('Recall@20(%)', fontweight='bold')
    # 添加双轴
    fig.legend(loc='upper center', frameon=False, ncol=5, handlelength=0.9,
               handleheight=0.9, fontsize='small')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.tick_params(labelsize=8)
    plt.show()


def intervention_vs_drop(dataset):
    x = ["2", "5", "10", "15", "20"]
    if dataset == 'Ciao':
        inter = [0.0837 * 100, 0.0816 * 100, 0.0809 * 100, 0.0794 * 100, 0.0805 * 100]
        drop = [0.0779 * 100, 0.0804 * 100, 0.0809 * 100, 0.0815 * 100, 0.0799 * 100]
    elif dataset == 'yahoo':
        inter = [0.1685 * 100, 0.1672 * 100, 0.1649 * 100, 0.1655 * 100, 0.1569 * 100]
        drop = [0.1642 * 100, 0.1636 * 100, 0.1586 * 100, 0.1643 * 100, 0.1598 * 100]
    elif dataset == 'coat':
        inter = [0.1641 * 100, 0.1678 * 100, 0.1635 * 100, 0.1636 * 100, 0.1629 * 100]
        drop = [0.1416 * 100, 0.1521 * 100, 0.1567 * 100, 0.1605 * 100, 0.1534 * 100]
    elif dataset == 'lastfm':
        inter = [0.3344 * 100, 0.3300 * 100, 0.3201 * 100, 0.3285 * 100, 0.3034 * 100]
        drop = [0.3330 * 100, 0.3330 * 100, 0.3224 * 100, 0.3269 * 100, 0.3068 * 100]
    else:
        raise
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(x, inter, "s--", label="DSE-I")
    plt.plot(x, drop, "s--", label="DSE-R")
    # for a, b in zip(x, inter):
    #     plt.text(a, b + 1, b, ha='center', va='bottom')
    #     # 数据显示的横坐标、显示的位置高度、显示的数据值的大小
    # for a, b in zip(x, drop):
    #     plt.text(a, b - 2, b, ha='center', va='bottom')

    # 绘图风格设置,使用seaborn库的API来设置样式
    # # 设置字体

    # x轴刻度标签设置
    plt.xticks(x)
    # y轴标签数值范围设置
    # 标题设置
    plt.xlabel("Ratio(%)")
    plt.ylabel("Recall@20(%)", fontweight='bold')
    plt.tick_params(labelsize=8)
    # 图例设置
    plt.legend()

    axes = plt.gca()
    pos = axes.get_position()
    new_pos = [pos.x0 + 0.02, pos.y0 + 0.05, pos.width, pos.height]
    axes.set_position(new_pos)
    plt.show()


def strcuture_loss(dataset):
    # 设置字体, 解决中文乱码问题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 解决图像中的'-'负号的乱码问题
    plt.rcParams['axes.unicode_minus'] = False

    x_labels = ['0.001', '0.01', '0.1', '1']
    if dataset == 'Ciao':
        recall = [0.0821 * 100, 0.0828 * 100, 0.0849 * 100, 0.0786 * 100]
        ndcg = [0.0551 * 100, 0.0552 * 100, 0.0553 * 100, 0.0487 * 100]
    elif dataset == 'yahoo':
        recall = [0.1695 * 100, 0.711 * 100, 0.1682 * 100, 0.1669 * 100]
        ndcg = [0.0810 * 100, 0.0804 * 100, 0.0789 * 100, 0.0791 * 100]
    else:
        raise
    legend_labels = ['Recall@20', 'NDCG@20']
    y = [recall, ndcg]
    fig = plt.figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(facecolor='white')
    # 红橙黄绿青蓝紫
    # color_list = ['#FF0000', '#FF8C00', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#800080']
    x_loc = np.arange(4)
    # x轴上每个刻度上能容纳的柱子的总的宽度设为0.8
    total_width = 0.8
    # 由y值可以看出x轴每个刻度上一共有3组数据, 也即3个柱子
    total_num = 2
    # 每个柱子的宽度用each_width表示
    each_width = total_width / total_num
    if total_num % 2 == 0:
        x1 = x_loc - (total_num / 2 - 1) * each_width - each_width / 2
    else:
        x1 = x_loc - ((total_num - 1) / 2) * each_width
    x_list = [x1 + each_width * i for i in range(total_num)]
    print(x_list)
    colors = ['darkorange', 'darkviolet']
    # 这里颜色设置成 橙色:"#FF8C00"
    for i in range(0, len(y)):
        ax.bar(x_list[i], y[i], width=each_width, color=colors[i])
    ax.set_xticks(x_loc)
    ax.set_xticklabels(x_labels)
    # ax.grid(True, ls=':', color='b', alpha=0.3)
    ax.set_xlabel(r'Structure loss weight $\lambda_S$')
    ax.set_ylabel('Metrics@20(%)', fontweight='bold')
    # 添加双轴
    # fig.legend(legend_labels, loc='upper left', bbox_to_anchor=(0.18, 0.95), fontsize=8, markerscale=5)
    # fig.legend(legend_labels, loc='upper left', bbox_to_anchor=(0., 0.95), fontsize=8, markerscale=5)
    plt.ylim(4.5)
    fig.tight_layout()
    # fig.subplots_adjust(top=0.9)
    # plt.tick_params(labelsize=8)
    axes = plt.gca()
    pos = axes.get_position()
    new_pos = [pos.x0 - 0.02, pos.y0 - 0.05, pos.width, pos.height]
    axes.set_position(new_pos)
    plt.legend(legend_labels, frameon=False, bbox_to_anchor=(0.5, 1.05), loc='center', ncol=3, borderaxespad=0.)
    plt.show()


def prior_weight(dataset):
    # 设置字体, 解决中文乱码问题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 解决图像中的'-'负号的乱码问题
    plt.rcParams['axes.unicode_minus'] = False

    x_labels = ['0.001', '0.01', '0.1', '1']
    if dataset == 'Ciao':
        recall = [0.0840 * 100, 0.0849 * 100, 0.0818 * 100, 0.0817 * 100]
        ndcg = [0.0546 * 100, 0.0553 * 100, 0.0538 * 100, 0.0547 * 100]
    elif dataset == 'yahoo':
        recall = [0.1695 * 100, 0.711 * 100, 0.1682 * 100, 0.1669 * 100]
        ndcg = [0.0810 * 100, 0.0804 * 100, 0.0789 * 100, 0.0791 * 100]
    else:
        raise
    legend_labels = ['Recall@20', 'NDCG@20']
    y = [recall, ndcg]
    fig = plt.figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(facecolor='white')
    # 红橙黄绿青蓝紫
    # color_list = ['#FF0000', '#FF8C00', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#800080']
    x_loc = np.arange(4)
    # x轴上每个刻度上能容纳的柱子的总的宽度设为0.8
    total_width = 0.8
    # 由y值可以看出x轴每个刻度上一共有3组数据, 也即3个柱子
    total_num = 2
    # 每个柱子的宽度用each_width表示
    each_width = total_width / total_num
    if total_num % 2 == 0:
        x1 = x_loc - (total_num / 2 - 1) * each_width - each_width / 2
    else:
        x1 = x_loc - ((total_num - 1) / 2) * each_width
    x_list = [x1 + each_width * i for i in range(total_num)]
    print(x_list)
    colors = ['darkorange', 'darkviolet']
    # 这里颜色设置成 橙色:"#FF8C00"
    for i in range(0, len(y)):
        ax.bar(x_list[i], y[i], width=each_width, color=colors[i])
    ax.set_xticks(x_loc)
    ax.set_xticklabels(x_labels)
    # ax.grid(True, ls=':', color='b', alpha=0.3)
    ax.set_xlabel(r'Fusion factor $\alpha$')
    ax.set_ylabel('Metrics@20(%)', fontweight='bold')
    # 添加双轴
    # fig.legend(legend_labels, loc='upper left', bbox_to_anchor=(0.18, 0.95), fontsize=8, markerscale=5)
    # fig.legend(legend_labels, loc='upper left', bbox_to_anchor=(0., 0.95), fontsize=8, markerscale=5)
    plt.ylim(5)
    fig.tight_layout()
    # fig.subplots_adjust(top=0.9)
    # plt.tick_params(labelsize=8)
    axes = plt.gca()
    pos = axes.get_position()
    new_pos = [pos.x0 - 0.02, pos.y0 - 0.05, pos.width, pos.height]
    axes.set_position(new_pos)
    plt.legend(legend_labels, frameon=False, bbox_to_anchor=(0.5, 1.05), loc='center', ncol=3, borderaxespad=0.)
    plt.show()


def intro_motivation(dataset):
    x = [10, 20, 30, 40, 50]
    if dataset == 'Ciao':
        # GCN = [0.7, 0.5, 1.8, 3.9, 5.5]
        # RecNS = [5.2, 5.2, 5.4, 14.6, 11.2]
        GCN = [5, 50.3, 57.8, 12.6, 35.2]
        RecNS = [18.6, 52.7, 26.4, 7.9, 18.3]
    else:
        raise
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(x, x, "r*--", label="threshold")
    plt.plot(x, GCN, "s-", label="GCN")
    plt.plot(x, RecNS, "bs-", label="GCN+RecNS")

    # for a, b in zip(x, inter):
    #     plt.text(a, b + 1, b, ha='center', va='bottom')
    #     # 数据显示的横坐标、显示的位置高度、显示的数据值的大小
    # for a, b in zip(x, drop):
    #     plt.text(a, b - 2, b, ha='center', va='bottom')

    # 绘图风格设置,使用seaborn库的API来设置样式
    # # 设置字体

    # x轴刻度标签设置
    plt.xticks(x)
    # y轴标签数值范围设置
    # 标题设置
    plt.xlabel("Ratio(%)")
    plt.ylabel(r"$\Delta Rec@20/Rec@20$(%)", fontweight='bold')
    plt.tick_params(labelsize=8)
    # 图例设置
    plt.legend(prop={'size': 8})
    # plt.legend(frameon=False, bbox_to_anchor=(0.5, 1.05), loc='center', ncol=3, borderaxespad=0.)

    axes = plt.gca()
    pos = axes.get_position()
    new_pos = [pos.x0 + 0.02, pos.y0 + 0.05, pos.width, pos.height]
    axes.set_position(new_pos)
    # plt.gca().set_aspect(0.6)
    plt.show()


if __name__ == '__main__':
    # structure_ablation()
    # interference_ablation()
    # fusion_negative_ablation()
    # intervention_vs_drop(dataset='coat')
    # extract_metrics()
    # strcuture_loss(dataset='Ciao')
    # prior_weight(dataset='Ciao')
    intro_motivation('Ciao')

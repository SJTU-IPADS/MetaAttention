# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.ticker as ticker
import argparse
from matplotlib.ticker import MultipleLocator
import os
import csv
from plot_utils import load_csv_data

colormap = plt.cm.Set2# LinearSegmentedColormap



def plot_figure11(results_dir, output_path="figure11_h100.pdf"):
    
    # load data for figure11
    deepseek_fwd_providers, deepseek_fwd_times_data = load_csv_data(
        os.path.join(results_dir, 'deepseek_fwd.csv'), 
        exclude_kv=False
    )
    deepseek_bwd_providers, deepseek_bwd_times_data = load_csv_data(
        os.path.join(results_dir, 'deepseek_bwd.csv'), 
        exclude_kv=True
    )
    
    llama_fwd_providers, llama_fwd_times_data = load_csv_data(
        os.path.join(results_dir, 'llama_fwd.csv'), 
        exclude_kv=False
    )
    
    llama_bwd_providers, llama_bwd_times_data = load_csv_data(
        os.path.join(results_dir, 'llama_bwd.csv'),
        exclude_kv=True
    )
    
    dit_fwd_providers, dit_fwd_times_data = load_csv_data(
        os.path.join(results_dir, 'dit_fwd.csv'),
        exclude_kv=False
    )
    
    dit_bwd_providers, dit_bwd_times_data = load_csv_data(
        os.path.join(results_dir, 'dit_bwd.csv'),
        exclude_kv=True
    )
    
    sigmoid_fwd_providers, sigmoid_fwd_times_data = load_csv_data(
        os.path.join(results_dir, 'sigmoid_attn_fwd.csv'),
        exclude_kv=False
    )
    
    sigmoid_bwd_providers, sigmoid_bwd_times_data = load_csv_data(
        os.path.join(results_dir, 'sigmoid_attn_bwd.csv'),
        exclude_kv=True
    )
    
    relu_fwd_providers, relu_fwd_times_data = load_csv_data(
        os.path.join(results_dir, 'vit_fwd.csv'),
        exclude_kv=False
    )
    
    relu_bwd_providers, relu_bwd_times_data = load_csv_data(
        os.path.join(results_dir, 'vit_bwd.csv'),
        exclude_kv=True
    )
    
    retnet_fwd_providers, retnet_fwd_times_data = load_csv_data(
        os.path.join(results_dir, 'retnet_fwd.csv'),
        exclude_kv=False
    )
    
    retnet_bwd_providers, retnet_bwd_times_data = load_csv_data(
        os.path.join(results_dir, 'retnet_recur_bwd.csv'),
        exclude_kv=True
    )
    
    mamba_fwd_providers, mamba_fwd_times_data = load_csv_data(
        os.path.join(results_dir, 'mamba2_fwd.csv'),
        exclude_kv=False
    )
    
    mamba_bwd_providers, mamba_bwd_times_data = load_csv_data(
        os.path.join(results_dir, 'mamba2_bwd.csv'),
        exclude_kv=True
    )
    
    gla_fwd_providers, gla_fwd_times_data = load_csv_data(
        os.path.join(results_dir, 'rfa_fwd.csv'),
        exclude_kv=False
    )
    
    gla_bwd_providers, gla_bwd_times_data = load_csv_data(
        os.path.join(results_dir, 'rfa_bwd.csv'),
        exclude_kv=True
    )
    
    gated_retnet_fwd_providers, gated_retnet_fwd_times_data = load_csv_data(
        os.path.join(results_dir, 'yoco_fwd.csv'),
        exclude_kv=False
    )
    
    gated_retnet_bwd_providers, gated_retnet_bwd_times_data = load_csv_data(
        os.path.join(results_dir, 'yoco_bwd.csv'),
        exclude_kv=True
    )
    
    retnet_chunk_fwd_providers, retnet_chunk_fwd_times_data = load_csv_data(
        os.path.join(results_dir, 'retnet_recur_fwd.csv'),
        exclude_kv=False
    )
    
    retnet_chunk_bwd_providers, retnet_chunk_bwd_times_data = load_csv_data(
        os.path.join(results_dir, 'retnet_recur_bwd.csv'),
        exclude_kv=True
    )
    
    mla_decode_fwd_providers, mla_decode_fwd_times_data = load_csv_data(
        os.path.join(results_dir, 'mla_fwd.csv'),
        exclude_kv=False
    )
    
    sparse_gqa_decode_fwd_providers, sparse_gqa_decode_fwd_times_data = load_csv_data(
        os.path.join(results_dir, 'sparse_gqa_fwd.csv'),
        exclude_kv=False
    )
    
    def combine(A, B):
        new_data = []
        names = []
        a_names = []
        b_names = []
        for a_item in A:
            a_names.append(a_item[0])
        for b_item in B:
            b_names.append(b_item[0])
        names = a_names.copy()
        # do not combine names
        # for b_name in b_names:
        #     if b_name not in names:
        #         names.append(b_name)
        a_len = len(A[0][1])
        b_len = len(B[0][1])
        new_data = []
        for name in names:
            if name in a_names:
                a_index = a_names.index(name)
                a_data = A[a_index][1]
            else:
                a_data = [0] * a_len
            if name in b_names:
                b_index = b_names.index(name)
                b_data = B[b_index][1]
            else:
                b_data = [0] * b_len
            data = a_data + b_data
            new_data.append((name, data))
        return new_data

    colers_sets = [
        (130 / 255, 176 / 255, 210 / 255),
        (146 / 255, 94 / 255, 176 / 255),
        (255 / 255, 190 / 255, 122 / 255),
        (250 / 255, 127 / 255, 111 / 255),
        (190 / 255, 184 / 255, 220 / 255),
        (231 / 255, 218 / 255, 210 / 255),
        (153 / 255, 153 / 255, 153 / 255),
        (150 / 255, 195 / 255, 125 / 255),
        (100 / 255, 100 / 255, 153 / 255),
        (200 / 255, 200 / 255, 153 / 255),
        (200 / 255, 177 / 255, 111 / 255),
        # nilu
        # (20 / 255, 54 / 255, 95 / 255),
        # (248 / 255, 231 / 255, 210 / 255),
        # # (118 / 255, 162 / 255, 185 / 255),
        # (191 / 255, 217 / 255, 229 / 255),
        # (214 / 255, 79 / 255, 56 / 255),
        # (112 / 255, 89 / 255, 146 / 255),
        # # dori
        # (214 / 255, 130 / 255, 148 / 255),
        # (169 / 255, 115 / 255, 153 / 255),
        # (248 / 255, 242 / 255, 236 / 255),
        # (214 / 255, 130 / 255, 148 / 255),
        # (243 / 255, 191 / 255, 202 / 255),
        # # (41/ 255, 31/ 255, 39/ 255),
        # # coller
        # # (72/ 255, 76/ 255, 35/ 255),
        # (124 / 255, 134 / 255, 65 / 255),
        # (185 / 255, 198 / 255, 122 / 255),
        # (248 / 255, 231 / 255, 210 / 255),
        # (182 / 255, 110 / 255, 151 / 255),
    ]

    # colers_sets = colormap(np.linspace(0, 1, 12))

    # 创建一个figure实例
    fig = plt.figure(figsize=(16, 14))

    # 获取Torch-Inductor的时间值
    _1x_baseline = "MetaAttention"

    fig = plt.figure(figsize=(15, 13))  # 宽度为12英寸，高度为10英寸
    # 设置网格布局
    gs = gridspec.GridSpec(6, 12, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1], wspace=0.3, hspace=0.9)

    hatch_patterns = ["-", "+", "x", "\\", "*", "o", "O", ".", "||", "//", "..."]

    legend_items = {}

    llm_legands = []
    other_legands = []


    def get_legend_item(label):
        if label not in legend_items:
            idx = len(legend_items)
            legend_items[label] = (
                colers_sets[idx % len(colers_sets)],
                hatch_patterns[idx % len(hatch_patterns)],
            )
        return legend_items[label]


    gs_llama = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 0:12], hspace=0.25)
    ax0_2 = fig.add_subplot(gs_llama[0])
    ax0_1 = fig.add_subplot(gs_llama[1:])
    ax0_2.set_ylim(4, 13)
    ax0_1.set_ylim(0, 3.0)
    ax0_2.axhline(y=1, color="black", linestyle="dashed")
    ax0_2.spines["bottom"].set_visible(False)
    ax0_2.set_xticklabels([])
    ax0_2.set_xticks([])
    ax0_1.spines["top"].set_visible(False)

    providers = deepseek_fwd_providers + deepseek_bwd_providers
    times_data = combine(deepseek_fwd_times_data, deepseek_bwd_times_data)
    _1x_baseline_times = dict(times_data)[_1x_baseline]
    norm_time_data = []
    for label, times in times_data:
        # if label != _1x_baseline:
        norm_time = [
            t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        norm_time_data.append((label, norm_time))

    # Create an array for x-axis positions
    x = np.arange(len(providers))

    # Set the width of the bars
    bar_width = 0.12

    # Draw cublas as a horizontal dashed line
    ax0_1.axhline(y=1, color="black", linestyle="dashed")

    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        ax0_2.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )

        ax0_1.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )

    ax0_1.plot(
        [2/3 - 1/64, 2/3 - 1/64],
        [1.5, -0.85],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax0_1.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    ax0_1.text(
        0.33,
        -0.88,
        "FWD",
        transform=ax0_1.transAxes,
        fontsize=12,
        ha="center",
    )

    ax0_1.text(
        0.80,
        -0.88,
        "BWD",
        transform=ax0_1.transAxes,
        fontsize=12,
        ha="center",
    )

    d = 0.01  # 斜线的长度
    kwargs = dict(transform=ax0_2.transAxes, color="k", clip_on=False)
    ax0_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
    ax0_2.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax0_1.transAxes)  # switch to the bottom axes
    ax0_1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax0_1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


    ax0_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax0_1.set_xticklabels(providers, fontsize=9)
    ax0_1.grid(False)


    gs_llama = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 0:7], hspace=0.25)
    ax1_2 = fig.add_subplot(gs_llama[0])
    ax1_1 = fig.add_subplot(gs_llama[1:])
    ax1_2.set_ylim(3, 14)  # 上面的图为10到最大值
    ax1_1.set_ylim(0, 2.0)  # 下面的图为0到5
    ax1_2.axhline(y=1, color="black", linestyle="dashed")
    ax1_2.spines["bottom"].set_visible(False)
    ax1_2.set_xticklabels([])
    ax1_2.set_xticks([])

    ax1_1.spines["top"].set_visible(False)

    providers = llama_fwd_providers + llama_bwd_providers
    times_data = combine(llama_fwd_times_data, llama_bwd_times_data)
    _1x_baseline_times = dict(times_data)[_1x_baseline]

    norm_time_data = []
    for label, times in times_data:
        # if label != _1x_baseline:
        norm_time = [
            t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        norm_time_data.append((label, norm_time))
    print(norm_time_data)
    x = np.arange(len(providers))
    bar_width =0.14
    ax1_1.axhline(y=1, color="black", linestyle="dashed")
    for i, (label, norm_time) in enumerate(norm_time_data):   
        rec = ax1_2.bar(
            x + i * bar_width,
            norm_time,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )

        rec = ax1_1.bar(
            x + i * bar_width,
            norm_time,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )


    ax1_1.plot(
        [0.502, 0.502],
        [1.5, -0.85],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax1_1.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    ax1_1.text(
        0.33,
        -0.88,
        "FWD",
        transform=ax1_1.transAxes,
        fontsize=12,
        ha="center",
    )

    ax1_1.text(
        0.80,
        -0.88,
        "BWD",
        transform=ax1_1.transAxes,
        fontsize=12,
        ha="center",
    )

    d = 0.01  # 斜线的长度
    kwargs = dict(transform=ax1_2.transAxes, color="k", clip_on=False)
    ax1_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
    ax1_2.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal


    kwargs.update(transform=ax1_1.transAxes)  # switch to the bottom axes
    ax1_1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax1_1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax1_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax1_1.set_xticklabels(providers, fontsize=10)
    ax1_1.grid(False)

    ax1_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax1_1.set_xticklabels(providers, fontsize=10)

    gs_llama = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[2, 0:8], hspace=0.25)
    ax2_2 = fig.add_subplot(gs_llama[0])
    ax2_1 = fig.add_subplot(gs_llama[1:])
    ax2_2.set_ylim(5, 10)  # 上面的图为10到最大值
    ax2_1.set_ylim(0, 4.0)  # 下面的图为0到5
    ax2_2.axhline(y=1, color="black", linestyle="dashed")
    ax2_2.spines["bottom"].set_visible(False)
    ax2_2.set_xticklabels([])
    ax2_2.set_xticks([])

    ax2_1.spines["top"].set_visible(False)

    providers = dit_fwd_providers + dit_bwd_providers
    times_data = combine(dit_fwd_times_data, dit_bwd_times_data)
    _1x_baseline_times = dict(times_data)[_1x_baseline]
    norm_time_data = []
    for label, times in times_data:
        # if label != _1x_baseline:
        norm_time = [
            t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        norm_time_data.append((label, norm_time))

    # Create an array for x-axis positions
    x = np.arange(len(providers))

    # Set the width of the bars
    bar_width = 0.12

    # Draw cublas as a horizontal dashed line
    ax2_1.axhline(y=1, color="black", linestyle="dashed")

    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        ax2_2.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )
        
        ax2_1.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )

    ax2_1.plot(
        [0.502, 0.502],
        [1.5, -0.85],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax2_1.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    ax2_1.text(
        0.33,
        -0.88,
        "FWD",
        transform=ax2_1.transAxes,
        fontsize=12,
        ha="center",
    )

    ax2_1.text(
        0.80,
        -0.88,
        "BWD",
        transform=ax2_1.transAxes,
        fontsize=12,
        ha="center",
    )

    d = 0.01  # 斜线的长度
    kwargs = dict(transform=ax2_2.transAxes, color="k", clip_on=False)
    ax2_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
    ax2_2.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal


    kwargs.update(transform=ax2_1.transAxes)  # switch to the bottom axes
    ax2_1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2_1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # X-axis and labels
    ax2_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax2_1.set_xticklabels(providers, fontsize=9)
    ax2_1.grid(False)

    gs_llama = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[3, 0:8], hspace=0.25)
    ax3_2 = fig.add_subplot(gs_llama[0])
    ax3_1 = fig.add_subplot(gs_llama[1:])
    ax3_2.set_ylim(4, 12)  # 上面的图为10到最大值
    ax3_1.set_ylim(0, 3.5)  # 下面的
    ax3_2.axhline(y=1, color="black", linestyle="dashed")
    ax3_2.spines["bottom"].set_visible(False)
    ax3_2.set_xticklabels([])
    ax3_2.set_xticks([])
    ax3_1.spines["top"].set_visible(False)

    providers = sigmoid_fwd_providers + sigmoid_bwd_providers
    times_data = combine(sigmoid_fwd_times_data, sigmoid_bwd_times_data)
    _1x_baseline_times = dict(times_data)[_1x_baseline]
    norm_time_data = []
    for label, times in times_data:
        # if label != _1x_baseline:
        norm_time = [
            t / p_i if p_i != 0 and t != 0 else 0
            for p_i, t in zip(_1x_baseline_times, times)
        ]
        norm_time_data.append((label, norm_time))

    # Create an array for x-axis positions
    x = np.arange(len(providers))

    # Set the width of the bars
    bar_width = 0.22

    # Draw cublas as a horizontal dashed line
    ax3_1.axhline(y=1, color="black", linestyle="dashed")

    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        rec = ax3_2.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )
        rec = ax3_1.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )
        for rect in rec:
            height = rect.get_height()
            if height == 0 and label == "PyTorch Inductor":
                warning_text = f"{label}\nFailed"
                ax3_1.text(
                    rect.get_x() + rect.get_width() / 2 + 0.01,
                    height + 0.05,
                    warning_text,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color="red",
                    weight="bold",
                )

    ax3_1.plot(
        [0.502, 0.502],
        [1.5, -0.85],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax3_1.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    ax3_1.text(
        0.33,
        -0.88,
        "FWD",
        transform=ax3_1.transAxes,
        fontsize=12,
        ha="center",
    )

    ax3_1.text(
        0.80,
        -0.88,
        "BWD",
        transform=ax3_1.transAxes,
        fontsize=12,
        ha="center",
    )

    d = 0.01  # 斜线的长度
    kwargs = dict(transform=ax3_2.transAxes, color="k", clip_on=False)
    ax3_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
    ax3_2.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax3_1.transAxes)  # switch to the bottom axes
    ax3_1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax3_1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # X-axis and labels
    ax3_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax3_1.set_xticklabels(providers, fontsize=9)
    ax3_1.grid(False)

    gs_llama = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[3, 8:12], hspace=0.25)
    ax4_2 = fig.add_subplot(gs_llama[0])
    ax4_1 = fig.add_subplot(gs_llama[1:])
    ax4_2.set_ylim(3, 10)  # 上面的图为10
    ax4_1.set_ylim(0, 2.0)  # 下面的图为0到5
    ax4_2.axhline(y=1, color="black", linestyle="dashed")
    ax4_2.spines["bottom"].set_visible(False)
    ax4_2.set_xticklabels([])
    ax4_2.set_xticks([])
    ax4_1.spines["top"].set_visible(False)

    providers = gla_fwd_providers + gla_bwd_providers
    times_data = combine(gla_fwd_times_data, gla_bwd_times_data)
    _1x_baseline_times = dict(times_data)[_1x_baseline]
    norm_time_data = []
    for label, times in times_data:
        # if label != _1x_baseline:
        norm_time = [
            t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        norm_time_data.append((label, norm_time))

    # Create an array for x-axis positions
    x = np.arange(len(providers))

    # Set the width of the bars
    bar_width = 0.25

    # Draw cublas as a horizontal dashed line
    ax4_1.axhline(y=1, color="black", linestyle="dashed")

    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        rec = ax4_2.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )
        rec = ax4_1.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )
        for rect in rec:
            height = rect.get_height()
            if height == 0:
                warning_text = f"FLA Failed"
                ax4_1.text(
                    rect.get_x() + rect.get_width() / 2 + 0.01,
                    height + 0.05,
                    warning_text,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color="red",
                    weight="bold",
                )

    ax4_1.plot(
        [0.502, 0.502],
        [1.5, -0.85],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax4_1.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    ax4_1.text(
        0.33,
        -0.88,
        "FWD",
        transform=ax4_1.transAxes,
        fontsize=12,
        ha="center",
    )

    ax4_1.text(
        0.80,
        -0.88,
        "BWD",
        transform=ax4_1.transAxes,
        fontsize=12,
        ha="center",
    )

    d = 0.01  # 斜线的长度
    kwargs = dict(transform=ax4_2.transAxes, color="k", clip_on=False)
    ax4_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
    ax4_2.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax4_1.transAxes)  # switch to the bottom axes
    ax4_1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax4_1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


    # X-axis and labels
    ax4_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax4_1.set_xticklabels(providers, fontsize=9)
    ax4_1.grid(False)

    ax5 = fig.add_subplot(gs[1, 7:12])  # ResNet
    providers = relu_fwd_providers + relu_bwd_providers
    times_data = combine(relu_fwd_times_data, relu_bwd_times_data)
    _1x_baseline_times = dict(times_data)[_1x_baseline]
    norm_time_data = []
    for label, times in times_data:
        # if label != _1x_baseline:
        norm_time = [
            t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        norm_time_data.append((label, norm_time))

    # Create an array for x-axis positions
    x = np.arange(len(providers))

    # Set the width of the bars
    bar_width = 0.25

    # Draw cublas as a horizontal dashed line
    ax5.axhline(y=1, color="black", linestyle="dashed")

    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        ax5.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )

    ax5.plot(
        [0.502, 0.502],
        [1.0, -0.6],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax5.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    ax5.text(
        0.33,
        -0.58,
        "FWD",
        transform=ax5.transAxes,
        fontsize=12,
        ha="center",
    )

    ax5.text(
        0.80,
        -0.58,
        "BWD",
        transform=ax5.transAxes,
        fontsize=12,
        ha="center",
    )

    # X-axis and labels
    ax5.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax5.set_xticklabels(providers, fontsize=9)
    ax5.grid(False)

    gs_llama = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[4, 0:8], hspace=0.25)
    ax6_2 = fig.add_subplot(gs_llama[0])
    ax6_1 = fig.add_subplot(gs_llama[1:])
    ax6_2.set_ylim(6, 18)  # 上面的图为10到最大值
    ax6_1.set_ylim(0, 4.0)  # 下面的图为0到5
    ax6_2.axhline(y=1, color="black", linestyle="dashed")
    ax6_2.spines["bottom"].set_visible(False)
    ax6_2.set_xticklabels([])
    ax6_2.set_xticks([])

    ax6_1.spines["top"].set_visible(False)

    providers = mamba_fwd_providers + mamba_bwd_providers
    times_data = combine(mamba_fwd_times_data, mamba_bwd_times_data)
    _1x_baseline_times = dict(times_data)[_1x_baseline]

    norm_time_data = []
    for label, times in times_data:
        # if label != _1x_baseline:
        norm_time = [
            t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        norm_time_data.append((label, norm_time))
    print(norm_time_data)
    x = np.arange(len(providers))
    bar_width =0.25
    ax6_1.axhline(y=1, color="black", linestyle="dashed")
    for i, (label, norm_time) in enumerate(norm_time_data):   
        rec = ax6_2.bar(
            x + i * bar_width,
            norm_time,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )

        rec = ax6_1.bar(
            x + i * bar_width,
            norm_time,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )


    ax6_1.plot(
        [0.502, 0.502],
        [1.5, -0.85],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax6_1.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    ax6_1.text(
        0.33,
        -0.88,
        "FWD",
        transform=ax6_1.transAxes,
        fontsize=12,
        ha="center",
    )

    ax6_1.text(
        0.80,
        -0.88,
        "BWD",
        transform=ax6_1.transAxes,
        fontsize=12,
        ha="center",
    )

    d = 0.01  # 斜线的长度
    kwargs = dict(transform=ax6_2.transAxes, color="k", clip_on=False)
    ax6_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
    ax6_2.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal


    kwargs.update(transform=ax6_1.transAxes)  # switch to the bottom axes
    ax6_1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax6_1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax6_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax6_1.set_xticklabels(providers, fontsize=10)
    ax6_1.grid(False)

    ax6_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax6_1.set_xticklabels(providers, fontsize=10)

    ax7 = fig.add_subplot(gs[2, 8:12])  # ResNet
    providers = retnet_fwd_providers + retnet_bwd_providers
    times_data = combine(retnet_fwd_times_data, retnet_bwd_times_data)
    _1x_baseline_times = dict(times_data)[_1x_baseline]
    norm_time_data = []
    for label, times in times_data:
        # if label != _1x_baseline:
        norm_time = [
            t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        norm_time_data.append((label, norm_time))

    # Create an array for x-axis positions
    x = np.arange(len(providers))

    # Set the width of the bars
    bar_width = 0.25

    # Draw cublas as a horizontal dashed line
    ax7.axhline(y=1, color="black", linestyle="dashed")

    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        ax7.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )

    ax7.plot(
        [0.502, 0.502],
        [1.0, -0.6],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax7.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    ax7.text(
        0.33,
        -0.58,
        "FWD",
        transform=ax7.transAxes,
        fontsize=12,
        ha="center",
    )

    ax7.text(
        0.80,
        -0.58,
        "BWD",
        transform=ax7.transAxes,
        fontsize=12,
        ha="center",
    )

    # X-axis and labels
    ax7.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax7.set_xticklabels(providers, fontsize=9)
    ax7.grid(False)

    ax8 = fig.add_subplot(gs[4, 8:12])  # ResNet
    providers = gated_retnet_fwd_providers + gated_retnet_bwd_providers
    times_data = combine(gated_retnet_fwd_times_data, gated_retnet_bwd_times_data)
    _1x_baseline_times = dict(times_data)[_1x_baseline]
    norm_time_data = []
    for label, times in times_data:
        # if label != _1x_baseline:
        norm_time = [
            t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        norm_time_data.append((label, norm_time))

    # Create an array for x-axis positions
    x = np.arange(len(providers))

    # Set the width of the bars
    bar_width = 0.25

    # Draw cublas as a horizontal dashed line
    ax8.axhline(y=1, color="black", linestyle="dashed")

    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        ax8.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )

    ax8.plot(
        [0.502, 0.502],
        [1.0, -0.6],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax8.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    ax8.text(
        0.33,
        -0.58,
        "FWD",
        transform=ax8.transAxes,
        fontsize=12,
        ha="center",
    )

    ax8.text(
        0.80,
        -0.58,
        "BWD",
        transform=ax8.transAxes,
        fontsize=12,
        ha="center",
    )

    # X-axis and labels
    ax8.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax8.set_xticklabels(providers, fontsize=9)
    ax8.grid(False)

    ax9 = fig.add_subplot(gs[5, 0:6])  # ResNet
    providers = retnet_chunk_fwd_providers + retnet_chunk_bwd_providers
    times_data = combine(retnet_chunk_fwd_times_data, retnet_chunk_bwd_times_data)
    _1x_baseline_times = dict(times_data)[_1x_baseline]
    norm_time_data = []
    for label, times in times_data:
        # if label != _1x_baseline:
        norm_time = [
            t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        norm_time_data.append((label, norm_time))

    # Create an array for x-axis positions
    x = np.arange(len(providers))

    # Set the width of the bars
    bar_width = 0.25

    # Draw cublas as a horizontal dashed line
    ax9.axhline(y=1, color="black", linestyle="dashed")

    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        ax9.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )

    ax9.plot(
        [0.502, 0.502],
        [1.0, -0.6],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax9.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    ax9.text(
        0.33,
        -0.58,
        "FWD",
        transform=ax9.transAxes,
        fontsize=12,
        ha="center",
    )

    ax9.text(
        0.80,
        -0.58,
        "BWD",
        transform=ax9.transAxes,
        fontsize=12,
        ha="center",
    )

    # X-axis and labels
    ax9.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax9.set_xticklabels(providers, fontsize=9)
    ax9.grid(False)

    ax10 = fig.add_subplot(gs[5, 6:9])  # ResNet
    providers = mla_decode_fwd_providers
    times_data = mla_decode_fwd_times_data
    _1x_baseline_times = dict(times_data)[_1x_baseline]
    norm_time_data = []
    for label, times in times_data:
        # if label != _1x_baseline:
        norm_time = [
            t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        norm_time_data.append((label, norm_time))
        
    # Create an array for x-axis positions
    x = np.arange(len(providers))

    # Set the width of the bars
    bar_width = 0.1

    # Draw cublas as a horizontal dashed line
    ax10.axhline(y=1, color="black", linestyle="dashed")
    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        ax10.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )
        
    ax10.text(
        0.52,
        -0.58,
        "FWD",
        transform=ax10.transAxes,
        fontsize=12,
        ha="center",
    )
        
    # X-axis and labels
    ax10.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax10.set_xticklabels(providers, fontsize=9)
    ax10.grid(False)

    ax11 = fig.add_subplot(gs[5, 9:12])  # ResNet
    providers = sparse_gqa_decode_fwd_providers
    times_data = sparse_gqa_decode_fwd_times_data
    _1x_baseline_times = dict(times_data)[_1x_baseline]
    norm_time_data = []
    for label, times in times_data:
        # if label != _1x_baseline:
        norm_time = [
            t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
        ]
        norm_time_data.append((label, norm_time))
        
    # Create an array for x-axis positions
    x = np.arange(len(providers))
    # Set the width of the bars
    bar_width = 0.1
    # Draw cublas as a horizontal dashed line
    ax11.axhline(y=1, color="black", linestyle="dashed")
    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        ax11.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )

    ax11.text(
        0.52,
        -0.58,
        "FWD",
        transform=ax11.transAxes,
        fontsize=12,
        ha="center",
    )
    # X-axis and labels
    ax11.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax11.set_xticklabels(providers, fontsize=9)
    ax11.grid(False)

        

    legand_font = 14
    ax0_2.text(
        0.5,
        1.06,
        "(a) Softmax Attention (DeepSeek-V2-Lite)",
        transform=ax0_2.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )

    ax1_2.text(
        0.5,
        1.66,
        "(b) Softmax Attention (LLAMA-3.1-8B)",
        transform=ax1_1.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )

    ax2_2.text(
        0.5,
        1.06,
        "(d) Softmax Attention (Diff-Transformer-3B)",
        transform=ax2_2.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )

    ax3_2.text(
        0.5,
        1.06,
        "(f) Sigmoid Attention (LLAMA-3.1-8B)",
        transform=ax3_2.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )

    ax4_2.text(
        0.5,
        1.06,
        "(h) Gated Retention (RFA-Big)",
        transform=ax4_2.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )

    ax5.text(
        0.5,
        1.06,
        "(c) ReLU Attention (ViT-s/16-style)",
        transform=ax5.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )

    ax6_2.text(
        0.5,
        1.26,
        "(g) Mamba2 SSM (Mamba2-2.7B)",
        transform=ax6_2.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )

    ax7.text(
        0.5,
        1.06,
        "(e) Retention Parallel (RetNet-6.7B)",
        transform=ax7.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )

    ax8.text(
        0.5,
        1.06,
        "(i) Gated Retention (YOCO-13B)",
        transform=ax8.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )

    ax9.text(
        0.5,
        1.06,
        "(j) RetNet Recurrent (RetNet-6.7B)",
        transform=ax9.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )

    ax10.text(
        0.5,
        1.06,
        "(k) DeepSeek MLA",
        transform=ax10.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )

    ax11.text(
        0.5,
        1.06,
        "(l) Sparse Group Query Attention",
        transform=ax11.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )


    y_size = 8
    ax0_1.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax0_2.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax1_1.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax1_2.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax2_1.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax2_2.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax3_1.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax3_2.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax4_1.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax4_2.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax5.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax6_1.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax6_2.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax7.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax8.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax9.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
    ax10.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小

    # ax3_2.yaxis.set_major_locator(MultipleLocator(5))
    axes = [ax0_1, ax1_1, ax2_1, ax3_1, ax4_1, ax5, ax6_1, ax7, ax8, ax9, ax10, ax11]
    for ax in axes:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes = [ax1_2, ax6_2]
    for ax in axes:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3))


    # 为上面六个图添加图例
    handles_other = []
    labels_other = []
    handles_Ladder = []
    labels_Ladder = []
    for ax in [ax0_1, ax0_2, ax1_1, ax1_2, ax2_1, ax2_2, ax3_1, ax3_2, ax4_1, ax4_2, ax5, ax6_1, ax6_2, ax7, ax8, ax9, ax10, ax11]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label == _1x_baseline:
                label = "MetaAttention(ours)"
            if label not in (labels_other + labels_Ladder):
                if "Ladder" in label:
                    handles_Ladder.append(handle)
                    labels_Ladder.append(label)
                else:
                    handles_other.append(handle)
                    labels_other.append(label)
            else:
                pass
    handles_other.extend(handles_Ladder)
    labels_other.extend(labels_Ladder)
    fig.legend(
        handles_other,
        labels_other,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.980 - 0.04),
        ncol=len(labels_other) // 2,
        fontsize=14,
        frameon=True,
    )

    # 调整布局以避免图例被遮挡
    fig.text(
        0.09,
        0.5,
        "Normalized latency Vs. MetaAttention \n(lower is better)",
        fontsize=20,
        rotation=90,
        va="center",
        ha="center",
    )
    plt.subplots_adjust(top=0.85, bottom=0.15)
    # plt.show()
    plt.savefig(
        output_path,
        bbox_inches="tight",
    )
    

if __name__ == "__main__":
    RESULTS_DIR = "/AttentionEngine/results_20251124"
    plot_figure11(RESULTS_DIR, "h100_eval.pdf")
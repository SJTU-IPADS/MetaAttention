from plot_utils import load_csv_data

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.ticker as ticker
import argparse
from matplotlib.ticker import MultipleLocator
import os


colormap = plt.cm.Set2# LinearSegmentedColormap


def plot_figure14(results_dir, output_path="figure14_mi250.pdf"):
    
    deepseek_fwd_providers, deepseek_fwd_times_data = load_csv_data(
        os.path.join(results_dir, "deepseek_fwd.csv"),
        exclude_kv=False,
    )
    
    deepseek_bwd_providers, deepseek_bwd_times_data = load_csv_data(
        os.path.join(results_dir, "deepseek_bwd.csv"),
        exclude_kv=True,
    )
    
    reluattn_fwd_providers, reluattn_fwd_times_data = load_csv_data(
        os.path.join(results_dir, "vit_fwd.csv"),
        exclude_kv=False,
    )
    
    reluattn_bwd_providers, reluattn_bwd_times_data = load_csv_data(
        os.path.join(results_dir, "vit_bwd.csv"),
        exclude_kv=True,
    )
     
    mamba2_fwd_providers, mamba2_fwd_times_data = load_csv_data(
        os.path.join(results_dir, "mamba2_fwd.csv"),
        exclude_kv=False,
    )
    
    mamba2_bwd_providers, mamba2_bwd_times_data = load_csv_data(
        os.path.join(results_dir, "mamba2_bwd.csv"),
        exclude_kv=True,
    )
    
    retnet_chunk_fwd_providers, retnet_chunk_fwd_times_data = load_csv_data(
        os.path.join(results_dir, "retnet_recur_fwd.csv"),
        exclude_kv=False,
    )
    
    retnet_chunk_bwd_providers, retnet_chunk_bwd_times_data = load_csv_data(
        os.path.join(results_dir, "retnet_recur_bwd.csv"),
        exclude_kv=True,
    )
    
    mla_decode_fwd_providers, mla_decode_fwd_times_data = load_csv_data(
        os.path.join(results_dir, "mla_fwd.csv"),
        exclude_kv=False,
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
        for b_name in b_names:
            if b_name not in names:
                names.append(b_name)
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
    
    # 创建一个figure实例
    fig = plt.figure(figsize=(8, 12))
    
    # 获取Torch-Inductor的时间值
    _1x_baseline = "MetaAttention"
    
    # 设置网格布局
    gs = gridspec.GridSpec(5, 6, figure=fig, height_ratios=[1, 1, 1, 1, 1], wspace=0.3, hspace=0.9)

    hatch_patterns = ["-", "+", "x", "\\", "*", "o", "O", "."]
    
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
    
    ax0 = fig.add_subplot(gs[0, 0:6])
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
    ax0.axhline(y=1, color="black", linestyle="dashed")
    
    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        rec = ax0.bar(
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
                warning_text = f"fa2 Failed"
                ax0.text(
                    rect.get_x() + rect.get_width() / 2 + 0.03,
                    height + 0.05,
                    warning_text,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color="red",
                    weight="bold",
                )
        
        
    # 画分割线
    ax0.plot(
        [1/2 - 1/64, 1/2 - 1/64],
        [1.0, -0.6],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax0.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    # 标fwd,hwd
    ax0.text(
        0.25,
        -0.58,
        "FWD",
        transform=ax0.transAxes,
        fontsize=12,
        ha="center",
    )
    
    ax0.text(
        0.75,
        -0.58,
        "BWD",
        transform=ax0.transAxes,
        fontsize=12,
        ha="center",
    )

    ax0.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax0.set_xticklabels(providers, fontsize=9)
    ax0.grid(False)

    # ax1 = fig.add_subplot(gs[1, 0:6])
    # providers = deepseek_fwd_providers + deepseek_bwd_providers
    # times_data = combine(deepseek_fwd_times_data, deepseek_bwd_times_data)
    # _1x_baseline_times = dict(times_data)[_1x_baseline]
    # norm_time_data = []
    # for label, times in times_data:
    #     # if label != _1x_baseline:
    #     norm_time = [
    #         t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    #     ]
    #     norm_time_data.append((label, norm_time))

    # # Create an array for x-axis positions
    # x = np.arange(len(providers))
    
    # # Set the width of the bars
    # bar_width = 0.12

    # # Draw cublas as a horizontal dashed line
    # ax1.axhline(y=1, color="black", linestyle="dashed")
    
    # # Create bars using a loop
    # for i, (label, speedup) in enumerate(norm_time_data):
    #     if label not in other_legands:
    #         other_legands.append(label)
    #     ax1.bar(
    #         x + i * bar_width,
    #         speedup,
    #         bar_width,
    #         label=label,
    #         linewidth=0.8,
    #         edgecolor="black",
    #         hatch=get_legend_item(label)[1],
    #         color=get_legend_item(label)[0],
    #     )
        
    # # 画分割线
    # ax1.plot(
    #     [1/2 - 1/64, 1/2 - 1/64],
    #     [1.0, -0.6],
    #     color='black',
    #     linestyle='dashed',
    #     linewidth=1.5,
    #     transform=ax1.transAxes,   # 使用 Axes 坐标
    #     clip_on=False             # 绘制范围可以超出 Axes
    # )

    # # 标fwd,hwd
    # ax1.text(
    #     0.25,
    #     -0.58,
    #     "FWD",
    #     transform=ax1.transAxes,
    #     fontsize=12,
    #     ha="center",
    # )
    
    # ax1.text(
    #     0.75,
    #     -0.58,
    #     "BWD",
    #     transform=ax1.transAxes,
    #     fontsize=12,
    #     ha="center",
    # )

    # ax1.set_xticks(x + len(norm_time_data) * bar_width / 2)
    # ax1.set_xticklabels(providers, fontsize=9)
    # ax1.grid(False)

    ax2 = fig.add_subplot(gs[1, 0:6])
    providers = reluattn_fwd_providers + reluattn_bwd_providers
    times_data = combine(reluattn_fwd_times_data, reluattn_bwd_times_data)
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
    ax2.axhline(y=1, color="black", linestyle="dashed")
    
    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        ax2.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )
        
    # 画分割线
    ax2.plot(
        [1/2 - 1/64, 1/2 - 1/64],
        [1.0, -0.6],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax2.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    # 标fwd,hwd
    ax2.text(
        0.25,
        -0.58,
        "FWD",
        transform=ax2.transAxes,
        fontsize=12,
        ha="center",
    )
    
    ax2.text(
        0.75,
        -0.58,
        "BWD",
        transform=ax2.transAxes,
        fontsize=12,
        ha="center",
    )

    ax2.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax2.set_xticklabels(providers, fontsize=9)
    ax2.grid(False)

    ax3 = fig.add_subplot(gs[2, 0:6])
    providers = mamba2_fwd_providers + mamba2_bwd_providers
    times_data = combine(mamba2_fwd_times_data, mamba2_bwd_times_data)
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
    ax3.axhline(y=1, color="black", linestyle="dashed")
    
    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        ax3.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )
        
    # 画分割线
    ax3.plot(
        [1/2 - 1/64, 1/2 - 1/64],
        [1.0, -0.6],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax3.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    # 标fwd,hwd
    ax3.text(
        0.25,
        -0.58,
        "FWD",
        transform=ax3.transAxes,
        fontsize=12,
        ha="center",
    )
    
    ax3.text(
        0.75,
        -0.58,
        "BWD",
        transform=ax3.transAxes,
        fontsize=12,
        ha="center",
    )

    ax3.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax3.set_xticklabels(providers, fontsize=9)
    ax3.grid(False)

    ax4 = fig.add_subplot(gs[3, 0:4])
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
    bar_width = 0.12

    # Draw cublas as a horizontal dashed line
    ax4.axhline(y=1, color="black", linestyle="dashed")
    
    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        ax4.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )
        
    # 画分割线
    ax4.plot(
        [1/2 - 1/64, 1/2 - 1/64],
        [1.0, -0.6],
        color='black',
        linestyle='dashed',
        linewidth=1.5,
        transform=ax4.transAxes,   # 使用 Axes 坐标
        clip_on=False             # 绘制范围可以超出 Axes
    )

    # 标fwd,hwd
    ax4.text(
        0.25,
        -0.58,
        "FWD",
        transform=ax4.transAxes,
        fontsize=12,
        ha="center",
    )
    
    ax4.text(
        0.75,
        -0.58,
        "BWD",
        transform=ax4.transAxes,
        fontsize=12,
        ha="center",
    )

    ax4.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax4.set_xticklabels(providers, fontsize=9)
    ax4.grid(False)

    gs_llama = gridspec.GridSpecFromSubplotSpec(3, 40, subplot_spec=gs[3, 4:6], hspace=0.25) # 20: 为了空一些
    ax5_2 = fig.add_subplot(gs_llama[0,1:])
    ax5_1 = fig.add_subplot(gs_llama[1:,1:])
    ax5_2.set_ylim(100, 150)  # 上面的图为10到最大值
    ax5_1.set_ylim(0, 12.0)  # 下面的图为0到5
    ax5_2.axhline(y=1, color="black", linestyle="dashed")
    ax5_2.spines["bottom"].set_visible(False)
    ax5_2.set_xticklabels([])
    ax5_2.set_xticks([])

    ax5_1.spines["top"].set_visible(False)

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
    bar_width = 0.12
    # Draw cublas as a horizontal dashed line
    ax5_1.axhline(y=1, color="black", linestyle="dashed")
    # Create bars using a loop
    for i, (label, speedup) in enumerate(norm_time_data):
        if label not in other_legands:
            other_legands.append(label)
        rec = ax5_2.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )
        rec = ax5_1.bar(
            x + i * bar_width,
            speedup,
            bar_width,
            label=label,
            linewidth=0.8,
            edgecolor="black",
            hatch=get_legend_item(label)[1],
            color=get_legend_item(label)[0],
        )
        
    ax5_1.text(
        0.5,
        -0.88,
        "FWD",
        transform=ax5_1.transAxes,
        fontsize=12,
        ha="center",
    )

    
    d = 0.01  # 斜线的长度
    kwargs = dict(transform=ax5_2.transAxes, color="k", clip_on=False)
    ax5_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
    ax5_2.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal


    kwargs.update(transform=ax5_1.transAxes)  # switch to the bottom axes
    ax5_1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax5_1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax5_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax5_1.set_xticklabels(providers, fontsize=10)
    ax5_1.grid(False)

    ax5_1.set_xticks(x + len(norm_time_data) * bar_width / 2)
    ax5_1.set_xticklabels(providers, fontsize=10)

    ax5_2.tick_params(axis="y", which="both", pad=0)
    ax5_1.tick_params(axis="y", which="both", pad=0)

    legand_font = 14
    ax0.text(
        0.5,
        1.06,
        "(a) Softmax Attention (DeepSeek-V2-Lite)",
        transform=ax0.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )
    # ax1.text(
    #     0.5,
    #     1.06,
    #     "(b) Softmax Attention (Llama3.1-8B)",
    #     transform=ax1.transAxes,
    #     fontsize=legand_font,
    #     fontweight="bold",
    #     ha="center",
    # )
    ax2.text(
        0.5,
        1.06,
        "(b) Relu Attention (ViT-s/16-style)",
        transform=ax2.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )
    ax3.text(
        0.5,
        1.06,
        "(g) Mamba2 SSM (Mamba2-2.7B)",
        transform=ax3.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )
    ax4.text(
        0.5,
        1.06,
        "(j) RetNet Recurrent (RetNet-6.7B)",
        transform=ax4.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )
    ax5_2.text(
        0.5,
        1.26,
        "(k) DeepSeek MLA",
        transform=ax5_2.transAxes,
        fontsize=legand_font,
        fontweight="bold",
        ha="center",
    )
        

    # 为上面六个图添加图例
    handles_other = []
    labels_other = []
    handles_Ladder = []
    labels_Ladder = []
    for ax in [ax0, ax2, ax3, ax4, ax5_1]:
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
    fig.legend(
        handles_other,
        labels_other,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.980 - 0.06),
        ncol=len(labels_other)//2,
        fontsize=13,
        frameon=True,
    )
    # 调整布局以避免图例被遮挡
    fig.text(
        0.05,
        0.5,
        "Normalized latency Vs. MetaAttention \n(lower is better)",
        fontsize=20,
        rotation=90,
        va="center",
        ha="center",
    )
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

if __name__ == "__main__":
    RESULTS_DIR = "/cfy/results_mi250_20251123"
    plot_figure14(RESULTS_DIR, "mi250_eval.pdf")
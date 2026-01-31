import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---- 全局样式 ----
sns.set_theme(style="white")

plt.rcParams.update({
    'font.size':           21,
    'axes.titlesize':      27,
    'axes.labelsize':      24,
    'xtick.labelsize':     17,
    'ytick.labelsize':     21,
    'legend.fontsize':     16,
    'legend.title_fontsize':18
})

# ---- 原始数据 (Human Evaluator) ----
writing_orig = {
    ('AgentLab','Bio'): 3.625,
    ('TinyScientist','Bio'): 3.85,
    ('AgentLab','ML'): 3.925,
    ('TinyScientist','ML'): 3.925
}
idea_orig = {
    ('AgentLab','Bio'): 3.55,
    ('TinyScientist','Bio'): 3.7,
    ('AgentLab','ML'): 3.725,
    ('TinyScientist','ML'): 3.825
}

# ---- 分组 ----
groups = [
    ('ML',  'Writing'),
    ('ML',  'Idea'),
    ('Bio', 'Writing'),
    ('Bio', 'Idea'),
]
group_labels = [f"{m} ({d})" for (d,m) in groups]
models = ['TinyScientist', 'AgentLab']

# ---- 重排数据 ----
values = []
for (domain, metric) in groups:
    if metric == 'Writing':
        vals = [writing_orig[(model, domain)] for model in models]
    else:
        vals = [idea_orig[(model, domain)] for model in models]
    values.append(vals)
values = np.array(values)

# ---- 自定义配色 (橙色系) ----
# TinyScientist = 深橙，AgentLab = 浅橙
color_map = {
    ('ML', 'TinyScientist'): '#D95F02',
    ('ML', 'AgentLab'): '#FDB462',
    ('Bio', 'TinyScientist'): '#D95F02',
    ('Bio', 'AgentLab'): '#FDB462',
}

# ---- 画图 ----
x = np.arange(len(groups))
width = 0.36

fig, ax = plt.subplots(figsize=(7.2, 5.5))

for j, model in enumerate(models):
    bar_colors = [
        color_map[(domain, model)] for (domain, metric) in groups
    ]
    ax.bar(
        x + (j-0.5)*width,
        values[:, j],
        width,
        label=model,
        color=bar_colors,
        edgecolor='black',
        linewidth=0.8
    )

# ---- 数值标签 ----
for container in ax.containers:
    for bar in container:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}",
                    (bar.get_x() + bar.get_width()/2, h),
                    ha='center', va='bottom', fontsize=18,
                    xytext=(0,2), textcoords='offset points')

# ---- 坐标轴与标签 ----
ax.set_xticks(x)
ax.set_xticklabels(group_labels, rotation=0, ha='center')
ax.set_ylabel("Quality Score (Human)", labelpad=6)
ax.set_ylim(3.3, 4.3)
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

# ---- 图例 ----
ax.legend(
    title="Model",
    loc='upper right',
    frameon=False
)

plt.tight_layout()
png_path = "./human_eval_quality.png"
pdf_path = "./human_eval_quality.pdf"
plt.savefig(png_path, bbox_inches='tight', dpi=300)
plt.savefig(pdf_path, bbox_inches='tight')
plt.close()

print("Saved:", png_path, pdf_path)


import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['水印提示+常规样本准确率', '水印提示+水印样本准确率', '常规提示+常规样本准确率', '常规提示+水印样本准确率']
values = [94.16, 99.92, 94.44, 93.10]

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Source Han Sans SC']

# 绘制尖峰柱状图
x = np.arange(len(labels))
width = 0.6
fig, ax = plt.subplots()
bars = ax.bar(x, values, width, color=['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0'])

# 设置标题和标签
ax.set_title('不同提示和样本类型的准确率')
ax.set_ylabel('准确率 (%)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')

# 为每个柱状图添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height}%', ha='center', va='bottom')

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Data from Table 7
models = ['ResNet50', 'MobileNetV3Large', 'EfficientNetV2L', 'ConvNeXtBase']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Standard augmentation results
standard = {
    'ResNet50': [0.82, 0.82, 0.82, 0.81],
    'MobileNetV3Large': [0.81, 0.82, 0.81, 0.81],
    'EfficientNetV2L': [0.88, 0.88, 0.88, 0.88],
    'ConvNeXtBase': [0.90, 0.90, 0.90, 0.90]
}

# Color-based augmentation results
color_based = {
    'ResNet50': [0.85, 0.85, 0.84, 0.85],
    'MobileNetV3Large': [0.86, 0.86, 0.85, 0.86],
    'EfficientNetV2L': [0.90, 0.91, 0.90, 0.91],
    'ConvNeXtBase': [0.92, 0.93, 0.92, 0.92]
}

# Settings
plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")
bar_width = 0.08
group_gap = 0.4  # gap between models
metric_gap = 0.02  # gap between bars in a metric group
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Calculate positions
index = np.arange(len(models)) * (len(metrics)*bar_width*2 + group_gap)

# Plot bars
for i, metric in enumerate(metrics):
    std_pos = index + i*(bar_width*2 + metric_gap)
    cb_pos = std_pos + bar_width

    std_vals = [standard[model][i] for model in models]
    cb_vals = [color_based[model][i] for model in models]

    plt.bar(std_pos, std_vals, bar_width, color=colors[i], edgecolor='black', alpha=0.85,
            label=f'{metric} (Standard)' if i == 0 else "")
    plt.bar(cb_pos, cb_vals, bar_width, color=colors[i], edgecolor='black', alpha=0.85,
            hatch='//', label=f'{metric} (Color-Based)' if i == 0 else "")

# Formatting
plt.title('Model Performance Comparison Across Metrics and Augmentation Strategies',
          fontsize=16, pad=20)
plt.xlabel('Model Architecture', fontsize=13)
plt.ylabel('Score', fontsize=13)
plt.xticks(index + 3*(bar_width*2 + metric_gap)/2, models, fontsize=12)
plt.yticks(np.arange(0.75, 1.01, 0.05), fontsize=11)
plt.ylim(0.75, 1.0)

# Custom legend
legend_elements = [
    Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='Standard'),
    Patch(facecolor='gray', edgecolor='black', alpha=0.7, hatch='//', label='Color-Based'),
    *[Patch(facecolor=colors[i], label=metrics[i]) for i in range(len(metrics))]
]

plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left',
           title='Augmentation / Metric')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_performance_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

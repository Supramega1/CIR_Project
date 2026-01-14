import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)

def calculate_sus(row):
    """
    Calculates System Usability Scale (0-100) (answers from columns 9-18)
    """
    sus_answers = row.iloc[9:19].values.astype(float)

    odd_sum = np.sum(sus_answers[0::2] - 1)
    even_sum = np.sum(5 - sus_answers[1::2])

    return (odd_sum + even_sum) * 2.5


def calculate_nasa_tlx(row):
    """
    Calculates Raw NASA-TLX (0-100) (answers from columns 3-8)
    """
    mental = row.iloc[3]
    physical = row.iloc[4]
    temporal = row.iloc[5]
    success = row.iloc[6]
    success_inverted = 11 - success # inverted scale
    effort = row.iloc[7]
    frustration = row.iloc[8]

    raw_score = np.mean([mental, physical, temporal, success_inverted, effort, frustration])
    return (raw_score - 1) * (100 / 9)


df1 = pd.read_csv('experiment1-results.csv', sep=';')
df2 = pd.read_csv('experiment2-results.csv', sep=';')

for df in [df1, df2]:
    df['SUS_Score'] = df.apply(calculate_sus, axis=1)
    df['NASA_TLX_Score'] = df.apply(calculate_nasa_tlx, axis=1)

dependent_vars = [
    ('Task completion time (s)', 'Time (s)'),
    ('Error count', 'Error Rate'),
    ('SUS_Score', 'User Satisfaction (SUS)'),
    ('NASA_TLX_Score', 'Cognitive Load (NASA-TLX)')
]

print("EXPERIMENT 1: YumEye vs Traditional (T-Test)")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Experiment 1: Recipe Search Method Comparison', fontsize=16)

groups = df1['Group'].unique()
group_a = df1[df1['Group'] == groups[0]]
group_b = df1[df1['Group'] == groups[1]]

for i, (col_name, label) in enumerate(dependent_vars):

    t_stat, p_val = stats.ttest_ind(group_a[col_name], group_b[col_name], equal_var=False)

    print(f"Variable: {label}")
    print(f"  Mean {groups[0]}: {group_a[col_name].mean():.2f} | {groups[1]}: {group_b[col_name].mean():.2f}")
    print(f"  T-statistic: {t_stat:.4f}, p-value: {p_val:.4e}")
    print("-" * 20)

    ax = axes[i // 2, i % 2]
    sns.boxplot(data=df1, x='Group', y=col_name, ax=ax, palette="Set2")
    sns.stripplot(data=df1, x='Group', y=col_name, ax=ax, color='black', alpha=0.5, jitter=True)
    ax.set_title(f"{label}, p={p_val:.3f}", fontsize=12)
    ax.set_ylabel(label)
    ax.set_xlabel("")

plt.tight_layout()
plt.show()


print("EXPERIMENT 2: Modality Comparison (One-way ANOVA)")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Experiment 2: Interaction Modality Comparison', fontsize=16)

modality_col = 'Modality'
modalities = df2[modality_col].unique()

for i, (col_name, label) in enumerate(dependent_vars):
    samples = [df2[df2[modality_col] == mod][col_name] for mod in modalities]

    # One-way ANOVA
    f_stat, p_val = stats.f_oneway(*samples)

    print(f"Variable: {label}")
    for mod in modalities:
        print(f"  Mean {mod}: {df2[df2[modality_col] == mod][col_name].mean():.2f}")
    print(f"  F-statistic: {f_stat:.4f}, p-value: {p_val:.4e}")
    print("-" * 20)

    # Wykres
    ax = axes2[i // 2, i % 2]
    sns.boxplot(data=df2, x=modality_col, y=col_name, ax=ax, palette="viridis")
    sns.stripplot(data=df2, x=modality_col, y=col_name, ax=ax, color='black', alpha=0.5, jitter=True)
    ax.set_title(f"{label}, p={p_val:.3f}", fontsize=12)
    ax.set_ylabel(label)
    ax.set_xlabel("")

plt.tight_layout()
plt.show()
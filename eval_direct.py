import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(f'results/{}.csv')

accuracy = (df['true_labels'] == df['pred_labels']).mean()

classes = df['true_labels'].unique()
accuracy_by_class = {}

for cls in classes:
    mask = df['true_labels'] == cls
    if mask.sum() > 0:
        accuracy_by_class[cls] = (df[mask]['true_labels'] == df[mask]['pred_labels']).mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
colors = ['#2ecc71' if accuracy >= 0.5 else '#e74c3c']
bars1 = ax1.bar(['Acurácia Geral'], [accuracy * 100], color=colors, width=0.5, edgecolor='black', linewidth=1.5)
ax1.set_ylim(0, 100)
ax1.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
ax1.set_title('Acurácia Geral do Modelo', fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Baseline 50%')

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=12)

ax1.legend()

ax2 = axes[1]
classes_sorted = sorted(accuracy_by_class.keys())
accuracies = [accuracy_by_class[cls] * 100 for cls in classes_sorted]
colors2 = ['#3498db', '#e67e22', '#9b59b6', '#1abc9c'][:len(classes_sorted)]

bars2 = ax2.bar(range(len(classes_sorted)), accuracies, color=colors2, edgecolor='black', linewidth=1.5)
ax2.set_xticks(range(len(classes_sorted)))
ax2.set_xticklabels(classes_sorted, rotation=45, ha='right')
ax2.set_ylim(0, 100)
ax2.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
ax2.set_title('Acurácia por Classe', fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('model_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 50)
print("ESTATÍSTICAS DO MODELO")
print("=" * 50)
print(f"Acurácia Geral: {accuracy * 100:.2f}%")
print(f"\nTotal de predições: {len(df)}")
print(f"Predições corretas: {(df['true_labels'] == df['pred_labels']).sum()}")
print(f"Predições incorretas: {(df['true_labels'] != df['pred_labels']).sum()}")
print("\n" + "=" * 50)
print("ACURÁCIA POR CLASSE")
print("=" * 50)
for cls in classes_sorted:
    acc = accuracy_by_class[cls]
    count = (df['true_labels'] == cls).sum()
    correct = ((df['true_labels'] == cls) & (df['true_labels'] == df['pred_labels'])).sum()
    print(f"{cls:20s}: {acc * 100:6.2f}% ({correct}/{count})")
print("=" * 50)
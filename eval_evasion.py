import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config

df = pd.read_csv(f'results/{config.OUTFILE}.csv')

accuracy = (df['true_labels'] == df['pred_labels']).mean()

classes = df['true_labels'].unique()
accuracy_by_class = {}
class_counts = {}

for cls in classes:
    mask = df['true_labels'] == cls
    count = mask.sum()
    if count > 0:
        correct = ((df['true_labels'] == cls) & (df['true_labels'] == df['pred_labels'])).sum()
        accuracy_by_class[cls] = (correct / count)
        class_counts[cls] = (correct, count)

classes_sorted = sorted(accuracy_by_class.keys())

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
colors = ['#2ecc71' if accuracy >= 0.5 else '#e74c3c']
bars1 = ax1.bar(['Acurácia Geral'], [accuracy * 100], color=colors, width=0.5, edgecolor='black', linewidth=2)
ax1.set_ylim(0, 100)
ax1.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
ax1.set_title('Acurácia Geral do Modelo', fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Baseline 50%')

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=14)
ax1.legend()

ax2 = fig.add_subplot(gs[:, 1])
accuracies = [accuracy_by_class[cls] * 100 for cls in classes_sorted]

colors_palette = plt.cm.Set3(np.linspace(0, 1, len(classes_sorted)))
bars2 = ax2.barh(range(len(classes_sorted)), accuracies, color=colors_palette, edgecolor='black', linewidth=1.5)

ax2.set_yticks(range(len(classes_sorted)))
ax2.set_yticklabels(classes_sorted, fontsize=10)
ax2.set_xlim(0, 100)
ax2.set_xlabel('Acurácia (%)', fontsize=12, fontweight='bold')
ax2.set_title('Acurácia por Classe', fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.axvline(x=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

for i, (bar, cls) in enumerate(zip(bars2, classes_sorted)):
    width = bar.get_width()
    correct, total = class_counts[cls]
    ax2.text(width + 2, bar.get_y() + bar.get_height()/2.,
             f'{width:.1f}% ({correct}/{total})',
             ha='left', va='center', fontweight='bold', fontsize=9)

ax3 = fig.add_subplot(gs[1, 0])
class_totals = [class_counts[cls][1] for cls in classes_sorted]
bars3 = ax3.bar(range(len(classes_sorted)), class_totals, color=colors_palette, edgecolor='black', linewidth=1.5)
ax3.set_xticks(range(len(classes_sorted)))
ax3.set_xticklabels([cls.split()[1] if len(cls.split()) > 1 else cls for cls in classes_sorted],
                     rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('Quantidade de Amostras', fontsize=11, fontweight='bold')
ax3.set_title('Distribuição das Classes no Dataset', fontsize=13, fontweight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.savefig('evasion_model_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 70)
print("ESTATÍSTICAS DO MODELO - EVASION BASED CLARITY")
print("=" * 70)
print(f"Acurácia Geral: {accuracy * 100:.2f}%")
print(f"\nTotal de predições: {len(df)}")
print(f"Predições corretas: {(df['true_labels'] == df['pred_labels']).sum()}")
print(f"Predições incorretas: {(df['true_labels'] != df['pred_labels']).sum()}")
print("\n" + "=" * 70)
print("ACURÁCIA POR CLASSE")
print("=" * 70)
print(f"{'Classe':<30s} {'Acurácia':>10s} {'Corretas/Total':>15s}")
print("-" * 70)
for cls in classes_sorted:
    acc = accuracy_by_class[cls]
    correct, total = class_counts[cls]
    print(f"{cls:<30s} {acc * 100:>9.2f}% {correct:>6d}/{total:<6d}")
print("=" * 70)

print("\n" + "=" * 70)
print("CLASSES MAIS CONFUNDIDAS")
print("=" * 70)
confusion_pairs = {}
for idx, row in df.iterrows():
    if row['true_labels'] != row['pred_labels']:
        pair = (row['true_labels'], row['pred_labels'])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

# Top 5 confusões
top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
for (true_class, pred_class), count in top_confusions:
    print(f"{true_class:<25s} → {pred_class:<25s}: {count:>3d} vezes")
print("=" * 70)
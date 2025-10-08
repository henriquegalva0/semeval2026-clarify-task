import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os

# Carregar dados do CSV
file_path = 'results/albert-base-v2-evasion_based_clarity.csv'  # Ajuste o nome do arquivo conforme necessário

# Verificar se o arquivo existe
if not os.path.exists(file_path):
    print(f"Erro: Arquivo {file_path} não encontrado!")
    print("Verifique se o arquivo está na pasta 'results'")
    exit()

# Ler o CSV
df = pd.read_csv(file_path)

# Verificar se as colunas necessárias existem
required_columns = ['true_labels', 'pred_labels']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Erro: Colunas faltando no CSV: {missing_columns}")
    print(f"Colunas disponíveis: {list(df.columns)}")
    exit()

print(f"Arquivo carregado com sucesso: {len(df)} linhas")
print(f"Colunas disponíveis: {list(df.columns)}")

# Calcular acurácia geral
accuracy = accuracy_score(df['true_labels'], df['pred_labels'])

# Calcular acurácia por classe
classes = sorted(df['true_labels'].unique())
class_accuracy = {}

for class_name in classes:
    mask = df['true_labels'] == class_name
    if mask.sum() > 0:
        class_acc = accuracy_score(df[mask]['true_labels'], df[mask]['pred_labels'])
        class_accuracy[class_name] = class_acc

# Gerar relatório de classificação
report = classification_report(df['true_labels'], df['pred_labels'], output_dict=True)

# Criar visualização
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análise de Desempenho do Modelo de Classificação', fontsize=16, fontweight='bold')

# Gráfico 1: Acurácia Geral e por Classe
ax1.bar(['Acurácia Geral'] + list(class_accuracy.keys()), 
        [accuracy] + list(class_accuracy.values()), 
        color=['skyblue'] + ['lightcoral', 'lightgreen', 'gold', 'plum'][:len(class_accuracy)])
ax1.set_title('Acurácia Geral e por Classe')
ax1.set_ylabel('Acurácia')
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Adicionar valores nas barras
for i, v in enumerate([accuracy] + list(class_accuracy.values())):
    ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Gráfico 2: Matriz de Confusão
cm = confusion_matrix(df['true_labels'], df['pred_labels'], labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax2)
ax2.set_title('Matriz de Confusão')
ax2.set_xlabel('Rótulos Previstos')
ax2.set_ylabel('Rótulos Verdadeiros')

# Gráfico 3: Distribuição das Classes
true_counts = df['true_labels'].value_counts()
pred_counts = df['pred_labels'].value_counts()

x = np.arange(len(classes))
width = 0.35

ax3.bar(x - width/2, [true_counts.get(cls, 0) for cls in classes], width, label='Rótulos Verdadeiros', alpha=0.7)
ax3.bar(x + width/2, [pred_counts.get(cls, 0) for cls in classes], width, label='Rótulos Previstos', alpha=0.7)
ax3.set_title('Distribuição das Classes - Verdadeiro vs Previsto')
ax3.set_xlabel('Classes')
ax3.set_ylabel('Contagem')
ax3.set_xticks(x)
ax3.set_xticklabels(classes, rotation=45)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Gráfico 4: Métricas Detalhadas por Classe
metrics = ['precision', 'recall', 'f1-score']
metrics_data = {metric: [report[cls][metric] for cls in classes if cls in report] for metric in metrics}

x = np.arange(len(classes))
width = 0.25

for i, metric in enumerate(metrics):
    ax4.bar(x + i*width, metrics_data[metric], width, label=metric.capitalize(), alpha=0.8)

ax4.set_title('Métricas Detalhadas por Classe')
ax4.set_xlabel('Classes')
ax4.set_ylabel('Score')
ax4.set_xticks(x + width)
ax4.set_xticklabels(classes, rotation=45)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# Print dos resultados numéricos
print("=" * 60)
print("RESULTADOS DETALHADOS DA ACURÁCIA")
print("=" * 60)
print(f"Acurácia Geral: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nAcurácia por Classe:")
for class_name, acc in class_accuracy.items():
    print(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)")

print("\n" + "=" * 60)
print("RELATÓRIO DE CLASSIFICAÇÃO")
print("=" * 60)
print(classification_report(df['true_labels'], df['pred_labels']))

# Estatísticas adicionais
print("\n" + "=" * 60)
print("ESTATÍSTICAS ADICIONAIS")
print("=" * 60)
print(f"Total de amostras: {len(df)}")
print(f"Número de classes: {len(classes)}")
print(f"Classes: {', '.join(classes)}")

# Se existir a coluna is_truncated, mostrar estatísticas
if 'is_truncated' in df.columns:
    print(f"\nEstatísticas de Truncamento:")
    print(f"  TRUE: {df['is_truncated'].sum()} amostras")
    print(f"  FALSE: {len(df) - df['is_truncated'].sum()} amostras")
    
    # Acurácia por status de truncamento
    if df['is_truncated'].sum() > 0:
        accuracy_truncated = accuracy_score(
            df[df['is_truncated']]['true_labels'], 
            df[df['is_truncated']]['pred_labels']
        )
        print(f"  Acurácia (TRUE): {accuracy_truncated:.4f}")
    
    if (len(df) - df['is_truncated'].sum()) > 0:
        accuracy_not_truncated = accuracy_score(
            df[~df['is_truncated']]['true_labels'], 
            df[~df['is_truncated']]['pred_labels']
        )
        print(f"  Acurácia (FALSE): {accuracy_not_truncated:.4f}")
        
plt.show(block=True)  
plt.pause(0.1)
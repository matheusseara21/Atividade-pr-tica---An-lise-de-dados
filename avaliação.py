import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configurações
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("=== ANÁLISE DO DATASET BOSTON HOUSING ===\n")
print("Descrição do dataset: https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset")
print("="*60)

# Carregar o dataset
try:
    df = pd.read_csv('housing.csv')
    print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas")
    print(f"Colunas originais: {list(df.columns)}")
except:
    print("Erro: arquivo housing.csv não encontrado na pasta")
    print("Certifique-se de que o arquivo está na mesma pasta do script")
    exit()

# Padronizar nomes das colunas para maiúsculas
column_mapping = {}
for col in df.columns:
    column_mapping[col] = col.upper()

df = df.rename(columns=column_mapping)
print(f"Colunas padronizadas: {list(df.columns)}")

# 1. ANÁLISE DE CONCENTRAÇÃO E DISTRIBUIÇÃO DE COLUNAS NUMÉRICAS
print("\n" + "="*60)
print("1. ANÁLISE DE CONCENTRAÇÃO E DISTRIBUIÇÃO - COLUNAS NUMÉRICAS")
print("="*60)

numeric_columns = df.select_dtypes(include=[np.number]).columns
print(f"Colunas numéricas encontradas ({len(numeric_columns)}): {list(numeric_columns)}")

# Estatísticas descritivas completas
print("\nESTATÍSTICAS DESCRITIVAS COMPLETAS:")
print("-" * 50)

desc_stats = df[numeric_columns].describe()
desc_stats.loc['variancia'] = df[numeric_columns].var()
desc_stats.loc['coef_variacao'] = df[numeric_columns].std() / df[numeric_columns].mean()
desc_stats.loc['assimetria'] = df[numeric_columns].skew()
desc_stats.loc['curtose'] = df[numeric_columns].kurtosis()

# Formatar para melhor visualização
formatted_stats = desc_stats.round(3)
print(formatted_stats)

# Análise de concentração por coeficiente de variação
print("\nANÁLISE DE CONCENTRAÇÃO (COEFICIENTE DE VARIAÇÃO):")
print("-" * 50)

for col in numeric_columns:
    cv = desc_stats.loc['coef_variacao', col]
    if cv > 1.5:
        concentracao = "✅ ALTA DISPERSÃO"
    elif cv > 0.5:
        concentracao = "⚠️ MÉDIA DISPERSÃO" 
    else:
        concentracao = "📊 BAIXA DISPERSÃO"
    print(f"{col:10}: CV = {cv:6.3f} - {concentracao}")

# 2. ANÁLISE DA MODA DAS COLUNAS CATEGÓRICAS
print("\n" + "="*60)
print("2. ANÁLISE DA MODA - COLUNAS CATEGÓRICAS")
print("="*60)

categorical_columns = df.select_dtypes(include=['object']).columns

if len(categorical_columns) > 0:
    print(f"Colunas categóricas encontradas ({len(categorical_columns)}): {list(categorical_columns)}")
    
    for col in categorical_columns:
        moda = df[col].mode()
        freq_absoluta = df[col].value_counts().iloc[0]
        freq_relativa = (freq_absoluta / len(df)) * 100
        
        print(f"\n{col}:")
        print(f"  Moda: {moda.iloc[0] if len(moda) > 0 else 'N/A'}")
        print(f"  Frequência absoluta: {freq_absoluta}")
        print(f"  Frequência relativa: {freq_relativa:.1f}%")
        
        # Mostrar top 5 valores se houver variedade
        if df[col].nunique() > 1:
            top_5 = df[col].value_counts().head(5)
            print(f"  Top 5 valores: {dict(top_5)}")
else:
    print("Nenhuma coluna categórica encontrada no dataset.")
    print("Análise da moda não se aplica.")

# Verificar se CHAS pode ser considerada categórica
print("\nANÁLISE DA VARIÁVEL CHAS (considerada categórica):")
if 'CHAS' in df.columns:
    chas_counts = df['CHAS'].value_counts()
    print(f"CHAS = 0 (Não limita com rio): {chas_counts.get(0, 0)} casos ({chas_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"CHAS = 1 (Limita com rio): {chas_counts.get(1, 0)} casos ({chas_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"Moda: {df['CHAS'].mode().iloc[0]}")

# 3. GRÁFICOS PARA ANÁLISE DE DISTRIBUIÇÃO
print("\n" + "="*60)
print("3. ANÁLISE VISUAL DE DISTRIBUIÇÃO - GRÁFICOS DE QUARTIS")
print("="*60)

# Boxplots para todas as variáveis numéricas
n_cols = len(numeric_columns)
n_rows = (n_cols + 3) // 4  # Calcula número de linhas necessário

fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
axes = axes.ravel() if n_rows > 1 else axes

for i, col in enumerate(numeric_columns):
    if i < len(axes):
        # Boxplot
        df.boxplot(column=col, ax=axes[i])
        axes[i].set_title(f'{col}\n(Q1: {df[col].quantile(0.25):.2f}, Q3: {df[col].quantile(0.75):.2f})')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Adicionar linhas de quartis
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        axes[i].axhline(y=q1, color='r', linestyle='--', alpha=0.7, label=f'Q1: {q1:.2f}')
        axes[i].axhline(y=q3, color='g', linestyle='--', alpha=0.7, label=f'Q3: {q3:.2f}')
        axes[i].legend()

# Remover eixos vazios
for i in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('boxplots_todas_variaveis.png', dpi=300, bbox_inches='tight')
plt.show()

# Histogramas para variáveis principais
print("\nGerando histogramas para variáveis principais...")
vars_principais = ['CRIM', 'RM', 'LSTAT', 'MEDV', 'AGE', 'NOX']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(vars_principais):
    if col in df.columns and i < len(axes):
        axes[i].hist(df[col], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribuição de {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequência')
        
        # Adicionar linhas de média e mediana
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Média: {mean_val:.2f}')
        axes[i].axvline(median_val, color='green', linestyle='--', label=f'Mediana: {median_val:.2f}')
        axes[i].legend()

plt.tight_layout()
plt.savefig('histogramas_principais.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. ANÁLISE DE CORRELAÇÃO ENTRE PARES DE COLUNAS NUMÉRICAS
print("\n" + "="*60)
print("4. ANÁLISE DE CORRELAÇÃO ENTRE PARES DE COLUNAS NUMÉRICAS")
print("="*60)

# Matriz de correlação completa
correlation_matrix = df[numeric_columns].corr()

print("MATRIZ DE CORRELAÇÃO COMPLETA:")
print("-" * 40)

# Formatar matriz para melhor visualização
corr_formatted = correlation_matrix.round(3)
print(corr_formatted)

# Heatmap detalhado
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.3f', cbar_kws={"shrink": .8}, 
            annot_kws={'size': 8})
plt.title('MATRIZ DE CORRELAÇÃO - TODOS OS PARES DE VARIÁVEIS NUMÉRICAS', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('matriz_correlacao_completa.png', dpi=300, bbox_inches='tight')
plt.show()

# Análise detalhada das correlações significativas
print("\nCORRELAÇÕES SIGNIFICATIVAS ENTRE PARES (|r| > 0.5 e p < 0.05):")
print("-" * 70)

significant_pairs = []
for i in range(len(numeric_columns)):
    for j in range(i+1, len(numeric_columns)):
        col1, col2 = numeric_columns[i], numeric_columns[j]
        corr, p_value = stats.pearsonr(df[col1], df[col2])
        
        if abs(corr) > 0.5 and p_value < 0.05:
            direction = 'POSITIVA' if corr > 0 else 'NEGATIVA'
            strength = 'FORTE' if abs(corr) > 0.7 else 'MODERADA' if abs(corr) > 0.5 else 'FRACA'
            
            significant_pairs.append({
                'Par': f"{col1} - {col2}",
                'Correlação': corr,
                'Direção': direction,
                'Força': strength,
                'p-value': p_value
            })

# Ordenar por força da correlação
significant_pairs.sort(key=lambda x: abs(x['Correlação']), reverse=True)

for pair in significant_pairs:
    print(f"  {pair['Par']:25}: r = {pair['Correlação']:7.3f} ({pair['Direção']:8} - {pair['Força']:8}) p = {pair['p-value']:.6f}")

# Gráficos de dispersão para correlações fortes
print(f"\nGerando gráficos de dispersão para {min(6, len(significant_pairs))} correlações mais fortes...")

n_plots = min(6, len(significant_pairs))
n_rows_plots = (n_plots + 2) // 3
fig, axes = plt.subplots(n_rows_plots, 3, figsize=(18, 5*n_rows_plots))

if n_rows_plots == 1:
    axes = [axes] if n_plots == 1 else axes

axes = axes.ravel() if n_rows_plots > 1 else axes

for idx in range(n_plots):
    if idx < len(significant_pairs):
        pair = significant_pairs[idx]
        col1, col2 = pair['Par'].split(' - ')
        
        axes[idx].scatter(df[col1], df[col2], alpha=0.6, color='steelblue')
        axes[idx].set_xlabel(col1)
        axes[idx].set_ylabel(col2)
        
        # Linha de tendência
        z = np.polyfit(df[col1], df[col2], 1)
        p = np.poly1d(z)
        axes[idx].plot(df[col1], p(df[col1]), "r--", alpha=0.8, linewidth=2)
        
        axes[idx].set_title(f'{col1} vs {col2}\nr = {pair["Correlação"]:.3f} ({pair["Direção"]})', 
                          fontweight='bold')

# Remover eixos vazios
for idx in range(n_plots, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('correlacoes_fortes.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. RESUMO COMPLETO DAS ANÁLISES
print("\n" + "="*60)
print("5. RESUMO GERAL DAS ANÁLISES")
print("="*60)

print("\n📊 RESUMO ESTATÍSTICO:")
print(f"- Total de observações: {len(df)}")
print(f"- Variáveis numéricas: {len(numeric_columns)}")
print(f"- Variáveis categóricas: {len(categorical_columns)}")

print("\n🎯 PRINCIPAIS CORRELAÇÕES IDENTIFICADAS:")
if significant_pairs:
    for pair in significant_pairs[:5]:  # Top 5
        print(f"- {pair['Par']}: r = {pair['Correlação']:.3f} ({pair['Força'].lower()}, {pair['Direção'].lower()})")
else:
    print("- Nenhuma correlação forte identificada")

print("\n📈 VARIÁVEIS COM MAIOR DISPERSÃO:")
high_dispersion = []
for col in numeric_columns:
    cv = desc_stats.loc['coef_variacao', col]
    if cv > 1:
        high_dispersion.append((col, cv))

high_dispersion.sort(key=lambda x: x[1], reverse=True)
for col, cv in high_dispersion[:3]:
    print(f"- {col}: CV = {cv:.3f}")

print("\n🔍 INSIGHTS IMPORTANTES:")
if 'MEDV' in df.columns:
    # Correlações com MEDV
    medv_correlations = []
    for col in numeric_columns:
        if col != 'MEDV':
            corr, _ = stats.pearsonr(df[col], df['MEDV'])
            if abs(corr) > 0.5:
                medv_correlations.append((col, corr))
    
    medv_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("Fatores que mais influenciam o valor das casas (MEDV):")
    for col, corr in medv_correlations[:3]:
        effect = "aumentam" if corr > 0 else "reduzem"
        print(f"- {col}: r = {corr:.3f} ({effect} o valor)")

print(f"\n✅ ANÁLISE CONCLUÍDA!")
print("Arquivos gerados:")
print("   - boxplots_todas_variaveis.png (Análise de quartis)")
print("   - histogramas_principais.png (Distribuição)")
print("   - matriz_correlacao_completa.png (Correlações)")
print("   - correlacoes_fortes.png (Principais relações)")
# Análise do Dataset Boston Housing

## 📋 Descrição do Projeto
Análise estatística completa do dataset Boston Housing para atividade avaliativa do 1º Bimestre. O projeto investiga fatores que influenciam o valor de propriedades na região de Boston.

## 🎯 Objetivos
- Analisar concentração e distribuição de variáveis numéricas
- Identificar correlações entre atributos das propriedades
- Compreender fatores determinantes do valor dos imóveis
- Gerar insights baseados em evidências estatísticas

## 📊 Dataset
**Fonte:** [Kaggle - Boston Housing Dataset](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset)

**Características:**
- 506 observações
- 14 variáveis (13 preditores + 1 target)
- Dados de 1978 coletados pelo U.S Census Service

**Variáveis Principais:**
- `CRIM`: Taxa de criminalidade per capita
- `RM`: Número médio de quartos
- `LSTAT`: % população de status baixo
- `MEDV`: Valor médio das casas (variável target)
- `CHAS`: Limitação com o rio Charles (dummy)
- `NOX`: Concentração de óxidos nítricos

## 🛠️ Tecnologias Utilizadas
- **Python 3.x**
- **Pandas** - Manipulação de dados
- **NumPy** - Cálculos numéricos
- **Matplotlib/Seaborn** - Visualizações
- **SciPy** - Testes estatísticos

## 📁 Estrutura do Projeto
```
boston-housing-analysis/
│
├── avaliação.py              # Código principal da análise
├── housing.csv               # Dataset original
├── boston_housing_analise.csv # Dataset processado
├── boxplots_todas_variaveis.png
├── histogramas_principais.png
├── matriz_correlacao_completa.png
├── correlacoes_fortes.png
└── README.md
```

## 📈 Métodos Estatísticos Aplicados

### 1. Estatística Descritiva
- Medidas de tendência central (média, mediana, moda)
- Medidas de dispersão (variância, desvio padrão, CV)
- Assimetria e curtose
- Análise de quartis e outliers

### 2. Análise de Correlação
- Matriz de correlação de Pearson
- Testes de significância (p-value)
- Classificação por força e direção
- Gráficos de dispersão

### 3. Testes de Hipóteses
- Teste t para comparação de médias (CHAS)
- Correlações parciais
- Significância estatística (α = 0.05)

## 🔍 Principais Descobertas

### Correlações Fortes com VALOR (MEDV):
- **LSTAT**: -0.738 (Negativa Forte) ✅
- **RM**: 0.695 (Positiva Forte) ✅
- **PTRATIO**: -0.508 (Negativa Moderada) ✅

### Hipóteses Confirmadas:
1. **Localização no rio** → +6.35K no valor médio (p < 0.001)
2. **Mais quartos** → Maior valor (r = 0.695, p < 0.001)
3. **Baixo status socioeconômico** → Menor valor (r = -0.738, p < 0.001)

### Relações Ambientais:
- **Áreas industriais × Poluição**: r = 0.763
- **Distância do centro × Poluição**: r = -0.769

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
```

### Execução do Código
```bash
python avaliação.py
```

### Saída Esperada
1. **Análise estatística completa** no console
2. **Gráficos** salvos em PNG
3. **Dataset processado** em CSV

## 📋 Requisitos Atendidos

- [x] Utilização do arquivo `housing.csv`
- [x] Análise de concentração e distribuição de colunas numéricas
- [x] Análise da moda de colunas categóricas
- [x] Análise de correlação entre pares de colunas numéricas
- [x] Relato numérico das análises
- [x] Gráficos para análise de quartis
- [x] Hipóteses comparativas entre valores dos imóveis
- [x] Descrição de correlações (valor, direção, força, confirmação)
- [x] Apresentação em slides (máx. 10 páginas)
- [x] Código disponibilizado

## 📊 Resultados Gerados

### Arquivos de Saída:
- `boxplots_todas_variaveis.png` - Análise de quartis
- `histogramas_principais.png` - Distribuições
- `matriz_correlacao_completa.png` - Correlações
- `correlacoes_fortes.png` - Principais relações
- `boston_housing_analise.csv` - Dataset processado

### Métricas Chave:
- **13 variáveis** analisadas estatisticamente
- **91 pares** de correlações calculadas
- **21 correlações** significativas identificadas
- **3 hipóteses** validadas estatisticamente

## 👥 Autores
*Análise desenvolvida para atividade avaliativa de Data Science*

## 📄 Licença
Este projeto é para fins educacionais. O dataset Boston Housing é de uso público para pesquisa e educação.

---
**⚠️ Nota:** Esta análise foi desenvolvida exclusivamente para fins acadêmicos e educacionais.

# earthquake-ml-pipeline

> Pipeline completo de Machine Learning para classificação de risco e predição de magnitude de terremotos com MLflow no Databricks

![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-003366?style=for-the-badge&logo=delta&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## Sobre o projeto

Pipeline end-to-end de Machine Learning que utiliza **28.700 eventos sísmicos reais** da API USGS (13 meses de histórico) para treinar modelos de classificação de risco e regressão de magnitude. O pipeline cobre desde a coleta de dados e feature engineering até o registro de modelos no MLflow Registry e inferência em batch, tudo dentro do ecossistema Databricks com Unity Catalog.

---

## Arquitetura do pipeline

```
[USGS API — 13 meses histórico]
           │
           ▼
[00_data_collection.py]
  • 19 arquivos JSON → Volume UC
  • Bronze + Silver reprocessados
  • 28.700 eventos sísmicos
           │
           ▼
[01_feature_engineering.py]
  • 19 features engenheiradas
  • Feature Store → gold.feature_store
           │
           ▼
[02_exploratory_analysis.py]
  • EDA + correlações
  • Matriz de correlação
  • Qualidade das features
           │
           ▼
[03_model_training.py]
  • 5 modelos treinados
  • MLflow tracking automático
  • 3 classificadores + 2 regressores
           │
           ▼
[04_model_evaluation.py]
  • Matriz de confusão
  • Feature Importance
  • MLflow Registry (v1)
           │
           ▼
[05_batch_inference.py]
  • 28.700 predições
  • gold.ml_predictions
  • Acurácia por classe
```

---

## Objetivos do modelo

| Problema | Tipo | Target | Métrica principal |
|----------|------|--------|-------------------|
| Classificação de risco | Multiclasse (4 classes) | `risk_level_enc` | F1 Score |
| Predição de magnitude | Regressão | `magnitude` | R² / RMSE |

---

## Feature Engineering

| Feature | Técnica | Motivação |
|---------|---------|-----------|
| `hour_sin` / `hour_cos` | Encoding cíclico | Hora 23 e 0 são vizinhas |
| `month_sin` / `month_cos` | Encoding cíclico | Sazonalidade sísmica |
| `is_shallow` | Flag binária | Terremotos rasos causam mais danos |
| `is_subduction_zone` | Flag binária | Zonas de maior risco de tsunami |
| `geo_region_enc` | Label encoding | Região geográfica → numérico |
| `depth_log` | Log transform | Reduz skewness da profundidade |
| `abs_latitude` | Feature derivada | Proxy de zona sísmica |

**Total: 19 features** | **28.700 registros** | **0.1% nulos**

---

## Resultados

### Classificação — Risk Level (4 classes)

| Modelo | F1 Score | Accuracy |
|--------|----------|----------|
| 🥇 **RandomForest** | **0.9888** | **98.90%** |
| 🥈 DecisionTree | 0.9852 | 98.54% |
| 🥉 LogisticRegression | 0.9229 | 93.00% |

**Métricas por classe (RandomForest):**

| Classe | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| LOW (M<4) | 0.9959 | 0.9888 | 0.9924 | 2.238 |
| MEDIUM (M4-5) | 0.9914 | 0.9965 | 0.9939 | 3.130 |
| HIGH (M5-6) | 0.9532 | 0.9737 | 0.9633 | 418 |
| CRITICAL (M≥6) | 0.7500 | 0.5854 | 0.6575 | 41 |

### Regressão — Magnitude

| Modelo | R² | RMSE |
|--------|----|------|
| 🥇 **RandomForest** | **0.9881** | **0.0952** |
| 🥈 GBT | 0.9835 | 0.1121 |

---

## Feature Importance (Top 5)

```
sig             ████████████████████████████████████████████  44.5%
mag_type_enc    ██████████████████████                        22.2%
geo_region_enc  ████████████                                  11.8%
longitude       ██████                                         5.9%
dmin            █████                                          5.2%
```

> `sig` (significância USGS) é a feature mais preditiva — correlação 0.94 com magnitude confirmada no EDA.

---

## Estrutura do projeto

```
earthquake-ml-pipeline/
├── 00_data_collection.py       # Coleta histórica 13 meses USGS
├── 01_feature_engineering.py   # 19 features + Feature Store Delta
├── 02_exploratory_analysis.py  # EDA + correlações + 4 gráficos
├── 03_model_training.py        # 5 modelos com MLflow tracking
├── 04_model_evaluation.py      # Avaliação + Feature Importance + Registry
└── 05_batch_inference.py       # 28.700 predições → gold.ml_predictions
```

---

## MLflow Registry

```
earthquake-risk-classifier       v1  ✅  (RandomForestClassifier)
earthquake-magnitude-regressor   v1  ✅  (RandomForestRegressor)
```

Ambos os modelos são Spark ML Pipelines completos:
`VectorAssembler → StandardScaler → RandomForest`

---

## Limitações e melhorias futuras

### Class Imbalance — CRITICAL (60% accuracy)
Eventos M≥6 representam apenas **0.5% do dataset** (156 de 28.700 registros). Em produção, aplicaria:

```python
# Opções para melhorar detecção de eventos CRITICAL:

# 1. Class weights no RandomForest
RandomForestClassifier(weightCol="class_weight")

# 2. Threshold tuning — priorizar recall sobre precision
# Reduzir threshold de 0.5 para 0.3 na classe CRITICAL

# 3. Coleta de mais dados históricos (> 5 anos, M≥5.5)
# A API USGS tem dados desde 1900
```

### RMSE alto em M≥6 (0.562)
Eventos extremos têm natureza caótica — difícil de prever com features estáticas. Melhorias possíveis: features temporais de sequência (terremotos precursores), dados de GPS geodésico.

---

## Stack técnica

| Tecnologia | Uso |
|------------|-----|
| **Databricks Free Edition** | Ambiente Serverless AWS |
| **Unity Catalog** | Feature Store + Model Registry storage |
| **MLflow** | Experiment tracking + Model Registry |
| **Spark ML** | Pipeline, VectorAssembler, StandardScaler |
| **RandomForestClassifier** | Classificação multiclasse |
| **RandomForestRegressor** | Regressão de magnitude |
| **Delta Lake** | Feature Store + Predictions table |
| **USGS Earthquake API** | 28.700 eventos reais |

---

## Como reproduzir

### Pré-requisitos
- Conta no [Databricks Free Edition](https://www.databricks.com/try-databricks)
- Unity Catalog habilitado
- Projeto `earthquake-streaming-pipeline` executado (tabelas Bronze/Silver)

### Passo a passo

```bash
# 1. Execute a coleta histórica (demora ~5 min — respeita rate limit USGS)
00_data_collection.py

# 2. Crie as features
01_feature_engineering.py

# 3. Analise os dados (opcional mas recomendado)
02_exploratory_analysis.py

# 4. Treine os modelos
03_model_training.py

# 5. Avalie e registre no MLflow
04_model_evaluation.py

# 6. Rode a inferência em batch
05_batch_inference.py
```

### Configuração do MLflow para Serverless

```python
# Obrigatório para Databricks Serverless
import os
os.environ["MLFLOW_DFS_TMP"] = "/Volumes/earthquake_pipeline/bronze/mlflow_tmp"

# Criar o volume antes de usar:
spark.sql("CREATE VOLUME IF NOT EXISTS earthquake_pipeline.bronze.mlflow_tmp")
```

---

## Projetos relacionados

Este projeto faz parte de uma série de 2 projetos:

| # | Projeto | Skills |
|---|---------|--------|
| 1 | [earthquake-streaming-pipeline](../earthquake-streaming-pipeline) | Streaming, Auto Loader, Alertas |
| 2 | **earthquake-ml-pipeline** ← você está aqui | ML, MLflow, Feature Engineering |

---
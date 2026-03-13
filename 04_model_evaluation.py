# Databricks notebook source
import mlflow
import mlflow.spark
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    RegressionEvaluator
)
from pyspark.sql import functions as F

# COMMAND ----------

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor":  "#161b22",
    "axes.edgecolor":   "#30363d", "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor":  "#ffffff", "xtick.color":     "#8b949e",
    "ytick.color":      "#8b949e", "text.color":      "#c9d1d9",
    "grid.color":       "#21262d", "grid.linestyle":  "--",
    "grid.alpha":       0.5,       "font.family":     "monospace",
})

MLFLOW_TMP = "/Volumes/earthquake_pipeline/bronze/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP

EXPERIMENT_NAME = "/Users/{}/earthquake-ml-pipeline".format(
    spark.sql("SELECT current_user()").collect()[0][0]
)
mlflow.set_experiment(EXPERIMENT_NAME)

FEATURE_TABLE = "earthquake_pipeline.gold.feature_store"
ML_FEATURES = [
    "latitude", "longitude", "abs_latitude",
    "depth_km", "depth_log",
    "sig", "nst", "gap", "rms", "dmin",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_shallow", "is_subduction_zone",
    "geo_region_enc", "depth_class_enc", "mag_type_enc"
]
TARGET_CLASS = "risk_level_enc"
TARGET_REG   = "magnitude"

print("Configurações carregadas")

# COMMAND ----------

# Retreinar o melhor modelo e gerar predições completas

print("Retreinando RandomForest (melhor modelo)...\n")

df = spark.table(FEATURE_TABLE) \
          .dropna(subset=ML_FEATURES + [TARGET_CLASS, TARGET_REG]) \
          .withColumn(TARGET_CLASS, F.col(TARGET_CLASS).cast("double"))

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

assembler = VectorAssembler(
    inputCols=ML_FEATURES, outputCol="features_raw", handleInvalid="skip"
)
scaler = StandardScaler(
    inputCol="features_raw", outputCol="features",
    withMean=True, withStd=True
)

# Melhor modelo de classificação
rf_clf = RandomForestClassifier(
    labelCol=TARGET_CLASS, featuresCol="features",
    numTrees=100, maxDepth=10, seed=42
)
pipeline_clf = Pipeline(stages=[assembler, scaler, rf_clf])
model_clf    = pipeline_clf.fit(train_df)
pred_clf     = model_clf.transform(test_df)

# Melhor modelo de regressão
rf_reg = RandomForestRegressor(
    labelCol=TARGET_REG, featuresCol="features",
    numTrees=100, maxDepth=10, seed=42
)
pipeline_reg = Pipeline(stages=[assembler, scaler, rf_reg])
model_reg    = pipeline_reg.fit(train_df)
pred_reg     = model_reg.transform(test_df) \
                         .withColumnRenamed("prediction", "pred_magnitude")

print(f" Modelos treinados")
print(f"   Treino : {train_df.count():,} | Teste: {test_df.count():,}")

# COMMAND ----------

# Matriz de confusão + métricas por classe
print("Avaliando classificador...\n")

# Coletar predições
pred_pd = pred_clf.select(
    TARGET_CLASS, "prediction"
).toPandas()
pred_pd[TARGET_CLASS]   = pred_pd[TARGET_CLASS].astype(int)
pred_pd["prediction"]   = pred_pd["prediction"].astype(int)

# Matriz de confusão manual
classes    = [0, 1, 2, 3]
labels_str = ["LOW\n(M<4)", "MEDIUM\n(M4-5)", "HIGH\n(M5-6)", "CRITICAL\n(M≥6)"]
n          = len(classes)
conf_matrix = np.zeros((n, n), dtype=int)

for true, pred in zip(pred_pd[TARGET_CLASS], pred_pd["prediction"]):
    if true in classes and pred in classes:
        conf_matrix[true][pred] += 1

# Métricas por classe
print(f"{'Classe':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 55)
for i, label in enumerate(labels_str):
    tp = conf_matrix[i][i]
    fp = conf_matrix[:, i].sum() - tp
    fn = conf_matrix[i, :].sum() - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0
    support   = conf_matrix[i, :].sum()
    print(f"{label.replace(chr(10),' '):<20} {precision:>10.4f} "
          f"{recall:>10.4f} {f1:>10.4f} {support:>10,}")

# Plot matriz de confusão
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(conf_matrix, cmap="Blues")
ax.set_xticks(range(n)); ax.set_xticklabels(labels_str, fontsize=8)
ax.set_yticks(range(n)); ax.set_yticklabels(labels_str, fontsize=8)
ax.set_xlabel("Predito",  fontweight="bold")
ax.set_ylabel("Real",     fontweight="bold")
ax.set_title("Matriz de Confusão — RandomForest Classifier", fontweight="bold", pad=15)
for i in range(n):
    for j in range(n):
        val = conf_matrix[i][j]
        color = "white" if val > conf_matrix.max() * 0.5 else "#c9d1d9"
        ax.text(j, i, f"{val:,}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
plt.colorbar(im, ax=ax, label="Contagem")
plt.tight_layout()
plt.show()
print("Matriz de confusão gerada")

# COMMAND ----------

# Avaliação do Regressor + gráficos
print("Avaliando regressor...\n")

pred_reg_pd = pred_reg.select(
    TARGET_REG, "pred_magnitude"
).toPandas()

residuals = pred_reg_pd[TARGET_REG] - pred_reg_pd["pred_magnitude"]

rmse = np.sqrt((residuals**2).mean())
mae  = residuals.abs().mean()
r2   = 1 - (residuals**2).sum() / \
       ((pred_reg_pd[TARGET_REG] - pred_reg_pd[TARGET_REG].mean())**2).sum()

print(f"   RMSE : {rmse:.4f}")
print(f"   MAE  : {mae:.4f}")
print(f"   R²   : {r2:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Real vs Predito
sample = pred_reg_pd.sample(min(3000, len(pred_reg_pd)), random_state=42)
axes[0].scatter(sample[TARGET_REG], sample["pred_magnitude"],
                alpha=0.3, s=8, color="#58a6ff")
mn = pred_reg_pd[TARGET_REG].min()
mx = pred_reg_pd[TARGET_REG].max()
axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfeito")
axes[0].set_title("Real vs Predito", fontweight="bold")
axes[0].set_xlabel("Magnitude Real")
axes[0].set_ylabel("Magnitude Predita")
axes[0].legend(); axes[0].grid(True)
axes[0].text(0.05, 0.92, f"R²={r2:.4f}", transform=axes[0].transAxes,
             color="#3fb950", fontsize=11, fontweight="bold")

# Distribuição dos resíduos
axes[1].hist(residuals, bins=50, color="#58a6ff",
             edgecolor="#0d1117", linewidth=0.3, alpha=0.85)
axes[1].axvline(0,    color="#f78166", linestyle="--",
                linewidth=2, label="Zero")
axes[1].axvline(rmse, color="#ffa657", linestyle="--",
                linewidth=1.5, label=f"RMSE={rmse:.3f}")
axes[1].axvline(-rmse,color="#ffa657", linestyle="--", linewidth=1.5)
axes[1].set_title("Distribuição dos Resíduos", fontweight="bold")
axes[1].set_xlabel("Resíduo (Real - Predito)")
axes[1].set_ylabel("Frequência")
axes[1].legend(fontsize=8); axes[1].grid(True, axis="y")

# Erro por faixa de magnitude
bins   = [0, 3, 4, 5, 6, 10]
labels = ["<M3","M3-4","M4-5","M5-6","M≥6"]
pred_reg_pd["mag_bin"] = pd.cut(
    pred_reg_pd[TARGET_REG], bins=bins, labels=labels
)
err_by_bin = pred_reg_pd.groupby("mag_bin", observed=True).apply(
    lambda x: np.sqrt(((x[TARGET_REG] - x["pred_magnitude"])**2).mean())
).reset_index()
err_by_bin.columns = ["mag_bin","rmse"]

colors = ["#3fb950","#58a6ff","#ffa657","#f78166","#d2a8ff"]
bars = axes[2].bar(err_by_bin["mag_bin"].astype(str),
                   err_by_bin["rmse"],
                   color=colors[:len(err_by_bin)],
                   edgecolor="#0d1117", linewidth=0.5)
for bar, val in zip(bars, err_by_bin["rmse"]):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.001,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=9, color="#ffffff")
axes[2].set_title("RMSE por Faixa de Magnitude", fontweight="bold")
axes[2].set_xlabel("Faixa de Magnitude")
axes[2].set_ylabel("RMSE")
axes[2].grid(True, axis="y")

plt.suptitle("Avaliação — RandomForest Regressor",
             fontsize=14, fontweight="bold", color="#ffffff")
plt.tight_layout()
plt.show()
print("Gráficos de regressão gerados")

# COMMAND ----------

# DBTITLE 1,Cell 6: Feature Importance
# Feature Importance
print("Feature Importance...\n")

# Extrair feature importance do RandomForest
rf_model_clf = model_clf.stages[-1]
importances  = rf_model_clf.featureImportances.toArray()

fi_df = pd.DataFrame({
    "feature":    ML_FEATURES,
    "importance": importances
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
colors  = ["#f78166" if v > fi_df["importance"].quantile(0.75)
           else "#58a6ff" for v in fi_df["importance"]]
bars = ax.barh(fi_df["feature"], fi_df["importance"],
               color=colors, edgecolor="#0d1117", linewidth=0.4)
for bar, val in zip(bars, fi_df["importance"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8, color="#c9d1d9")
ax.set_title("Feature Importance — RandomForest Classifier",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Importância")
ax.grid(True, axis="x")
plt.tight_layout()
plt.show()

print("\n Top 5 features mais importantes:")
for _, row in fi_df.sort_values("importance", ascending=False).head(5).iterrows():
    bar = "*" * int(row["importance"] * 200)
    print(f"   {row['feature']:<25} {row['importance']:.4f}  {bar}")

# COMMAND ----------

# DBTITLE 1,Cell 7
# Registrar melhor modelo no MLflow Registry
print("Registrando melhor modelo no MLflow Registry...\n")

MODEL_NAME_CLF = "earthquake-risk-classifier"
MODEL_NAME_REG = "earthquake-magnitude-regressor"

# Prepare MLflow model signature and input_example
from mlflow.models.signature import infer_signature

test_pd = test_df.limit(1).toPandas().copy()
# Verifica se todas as ML_FEATURES estão presentes
input_example = test_pd[ML_FEATURES]
# Exemplo de resultado do classificador
output_example_clf = pd.DataFrame({"prediction": [0.0]})
# Exemplo de saída da regressão
output_example_reg = pd.DataFrame({"pred_magnitude": [0.0]})

# Inferir assinaturas
sig_clf = infer_signature(input_example, output_example_clf)
sig_reg = infer_signature(input_example, output_example_reg)

with mlflow.start_run(run_name="BestModel_RandomForest_FINAL") as run:

    # Métricas finais
    acc  = MulticlassClassificationEvaluator(labelCol=TARGET_CLASS, predictionCol="prediction", metricName="accuracy").evaluate(pred_clf)
    f1   = MulticlassClassificationEvaluator(labelCol=TARGET_CLASS, predictionCol="prediction", metricName="f1").evaluate(pred_clf)
    rmse = RegressionEvaluator(labelCol=TARGET_REG, predictionCol="pred_magnitude", metricName="rmse").evaluate(pred_reg)
    r2   = RegressionEvaluator(labelCol=TARGET_REG, predictionCol="pred_magnitude", metricName="r2").evaluate(pred_reg)

    # Log params e métricas
    mlflow.log_params({
        "model"       : "RandomForestClassifier + RandomForestRegressor",
        "num_features": len(ML_FEATURES),
        "num_trees"   : 100,
        "max_depth"   : 10,
        "train_size"  : train_df.count(),
        "test_size"   : test_df.count(),
        "data_period" : "Mar/2025 - Mar/2026"
    })
    mlflow.log_metrics({
        "clf_accuracy": acc,
        "clf_f1"      : f1,
        "reg_rmse"    : rmse,
        "reg_r2"      : r2
    })

    # Salvar modelos
    mlflow.spark.log_model(
        model_clf, "classifier",
        registered_model_name=MODEL_NAME_CLF,
        input_example=input_example,
        signature=sig_clf
    )
    mlflow.spark.log_model(
        model_reg, "regressor",
        registered_model_name=MODEL_NAME_REG,
        input_example=input_example,
        signature=sig_reg
    )

    run_id = run.info.run_id

print(f"""
MODELOS REGISTRADOS NO MLFLOW

   Classificador
     Accuracy : {acc:.4f}
     F1 Score : {f1:.4f}

   Regressor
     R²       : {r2:.4f}
     RMSE     : {rmse:.4f}

Run ID : {run_id[:32]}...

""")
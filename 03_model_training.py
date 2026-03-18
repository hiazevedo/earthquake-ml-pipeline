# Databricks notebook source
import os
import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import (
    RandomForestClassifier,
    GBTClassifier,
    LogisticRegression
)
from pyspark.ml.regression import (
    RandomForestRegressor,
    GBTRegressor
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    RegressionEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F

from pyspark.ml.classification import DecisionTreeClassifier

# COMMAND ----------

# Configuração crítica para Serverless
# MLflow no Serverless exige UC Volume como storage para artefatos
MLFLOW_TMP = "/Volumes/earthquake_pipeline/bronze/mlflow_tmp"

# Criar o diretório se não existir
dbutils.fs.mkdirs(MLFLOW_TMP)

# Setar a variável de ambiente que o MLflow usa internamente

os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP

# Experiment
EXPERIMENT_NAME = "/Users/{}/earthquake-ml-pipeline".format(
    spark.sql("SELECT current_user()").collect()[0][0]
)
mlflow.set_experiment(EXPERIMENT_NAME)

# Features
FEATURE_TABLE = "earthquake_pipeline.gold.feature_store"

ML_FEATURES = [
    "latitude", "longitude", "abs_latitude",
    "depth_km", "depth_log",
    "sig", "nst", "gap", "rms", "dmin",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "is_shallow", "is_subduction_zone",
    "geo_region_enc", "depth_class_enc", "mag_type_enc"
]

TARGET_CLASS = "risk_level_enc"
TARGET_REG   = "magnitude"

print(f"   MLflow configurado para Serverless")
print(f"   Storage : {MLFLOW_TMP}")
print(f"   Experiment: {EXPERIMENT_NAME}")
print(f"   Features  : {len(ML_FEATURES)}")

# COMMAND ----------

# Preparar dados: split treino/teste + tratar nulos

print("Preparando dados...\n")

df = spark.table(FEATURE_TABLE) \
          .dropna(subset=ML_FEATURES + [TARGET_CLASS, TARGET_REG]) \
          .withColumn(TARGET_CLASS,
                      F.col(TARGET_CLASS).cast("double"))

# Split 80/20
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

train_count = train_df.count()
test_count  = test_df.count()

print(f"Treino : {train_count:,} registros")
print(f"Teste  : {test_count:,} registros")
print(f"   Ratio  : {train_count/(train_count+test_count)*100:.1f}% / "
      f"{test_count/(train_count+test_count)*100:.1f}%")

print("\n Distribuição do target no treino:")
display(train_df.groupBy(TARGET_CLASS).count().orderBy(TARGET_CLASS))

# COMMAND ----------

# Pipeline de pré-processamento compartilhado

assembler = VectorAssembler(
    inputCols  = ML_FEATURES,
    outputCol  = "features_raw",
    handleInvalid = "skip"
)

scaler = StandardScaler(
    inputCol  = "features_raw",
    outputCol = "features",
    withMean  = True,
    withStd   = True
)

print("Pipeline de pré-processamento definido")
print(f"   VectorAssembler : {len(ML_FEATURES)} features → features_raw")
print(f"   StandardScaler  : features_raw → features (normalizado)")

# COMMAND ----------

# Treinar 3 modelos de classificação


print("=" * 60)
print("  TREINAMENTO — CLASSIFICAÇÃO (risk_level_enc)")
print("=" * 60)

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol      = TARGET_CLASS,
    predictionCol = "prediction",
    metricName    = "accuracy"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol      = TARGET_CLASS,
    predictionCol = "prediction",
    metricName    = "f1"
)

classificadores = {
    "RandomForest_Classifier": RandomForestClassifier(
        labelCol    = TARGET_CLASS,
        featuresCol = "features",
        numTrees    = 100,
        maxDepth    = 10,
        seed        = 42
    ),
    "DecisionTree_Classifier": DecisionTreeClassifier(
        labelCol    = TARGET_CLASS,
        featuresCol = "features",
        maxDepth    = 10,
        seed        = 42
    ),
    "LogisticRegression_Classifier": LogisticRegression(
        labelCol  = TARGET_CLASS,
        featuresCol = "features",
        maxIter   = 100,
        regParam  = 0.01,
        family    = "multinomial"
    )
}

resultados_clf = {}

for nome, modelo in classificadores.items():
    print(f"\n Treinando {nome}...")

    pipeline = Pipeline(stages=[assembler, scaler, modelo])

    with mlflow.start_run(run_name=nome) as run:
        pipeline_model = pipeline.fit(train_df)
        predictions    = pipeline_model.transform(test_df)

        accuracy = evaluator_acc.evaluate(predictions)
        f1       = evaluator_f1.evaluate(predictions)

        mlflow.log_param("model_type",   nome)
        mlflow.log_param("num_features", len(ML_FEATURES))
        mlflow.log_metric("accuracy",    accuracy)
        mlflow.log_metric("f1_score",    f1)
        mlflow.spark.log_model(pipeline_model, "model")

        run_id = run.info.run_id
        resultados_clf[nome] = {
            "accuracy": accuracy,
            "f1":       f1,
            "run_id":   run_id
        }

        print(f"   Accuracy : {accuracy:.4f}")
        print(f"   F1 Score : {f1:.4f}")
        print(f"   Run ID  : {run_id}")

print("\n Ranking dos modelos de classificação:")
for nome, res in sorted(resultados_clf.items(),
                        key=lambda x: x[1]["f1"], reverse=True):
    print(f"   {nome:<40} F1: {res['f1']:.4f} | Acc: {res['accuracy']:.4f}")

# COMMAND ----------

# Treinar 2 modelos de regressão com MLflow tracking

print("=" * 60)
print("  TREINAMENTO — REGRESSÃO (magnitude)")
print("=" * 60)

evaluator_rmse = RegressionEvaluator(
    labelCol      = TARGET_REG,
    predictionCol = "prediction",
    metricName    = "rmse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol      = TARGET_REG,
    predictionCol = "prediction",
    metricName    = "r2"
)

regressores = {
    "RandomForest_Regressor": RandomForestRegressor(
        labelCol    = TARGET_REG,
        featuresCol = "features",
        numTrees    = 100,
        maxDepth    = 10,
        seed        = 42
    ),
    "GBT_Regressor": GBTRegressor(
        labelCol    = TARGET_REG,
        featuresCol = "features",
        maxIter     = 50,
        maxDepth    = 8,
        seed        = 42
    )
}

resultados_reg = {}

for nome, modelo in regressores.items():
    print(f"\n Treinando {nome}...")

    pipeline = Pipeline(stages=[assembler, scaler, modelo])

    with mlflow.start_run(run_name=nome) as run:
        pipeline_model = pipeline.fit(train_df)
        predictions    = pipeline_model.transform(test_df)

        rmse = evaluator_rmse.evaluate(predictions)
        r2   = evaluator_r2.evaluate(predictions)

        mlflow.log_param("model_type",   nome)
        mlflow.log_param("num_features", len(ML_FEATURES))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2",   r2)
        mlflow.spark.log_model(pipeline_model, "model")

        run_id = run.info.run_id
        resultados_reg[nome] = {
            "rmse":   rmse,
            "r2":     r2,
            "run_id": run_id
        }

        print(f"   RMSE    : {rmse:.4f}")
        print(f"   R²      : {r2:.4f}")
        print(f"   Run ID : {run_id}")

print("\nRanking dos modelos de regressão:")
for nome, res in sorted(resultados_reg.items(),
                        key=lambda x: x[1]["r2"], reverse=True):
    print(f"   {nome:<40} R²: {res['r2']:.4f} | RMSE: {res['rmse']:.4f}")

# COMMAND ----------

# Resumo final dos experimentos

print("=" * 60)
print("  RESUMO FINAL — TODOS OS MODELOS")
print("=" * 60)

print("\n - CLASSIFICAÇÃO (risk_level_enc):")
print(f"{'Modelo':<40} {'F1':>8} {'Accuracy':>10} {'Run ID':>15}")
print("-" * 75)
for nome, res in sorted(resultados_clf.items(),
                        key=lambda x: x[1]["f1"], reverse=True):
    print(f"{nome:<40} {res['f1']:>8.4f} {res['accuracy']:>10.4f} "
          f"{res['run_id'][:8]:>15}")

print("\n - REGRESSÃO (magnitude):")
print(f"{'Modelo':<40} {'R²':>8} {'RMSE':>10} {'Run ID':>15}")
print("-" * 75)
for nome, res in sorted(resultados_reg.items(),
                        key=lambda x: x[1]["r2"], reverse=True):
    print(f"{nome:<40} {res['r2']:>8.4f} {res['rmse']:>10.4f} "
          f"{res['run_id'][:8]:>15}")

print(f"""
TREINAMENTO CONCLUÍDO
    Modelos treinados : 5 (3 clf + 2 reg)
    Experimentos MLflow registrados : 5 runs
""")
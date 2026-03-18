# Databricks notebook source
import mlflow
import mlflow.spark
import os
from pyspark.sql import functions as F

# COMMAND ----------

MLFLOW_TMP = "/Volumes/earthquake_pipeline/bronze/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP

EXPERIMENT_NAME = "/Users/{}/earthquake-ml-pipeline".format(
    spark.sql("SELECT current_user()").collect()[0][0]
)
mlflow.set_experiment(EXPERIMENT_NAME)

FEATURE_TABLE     = "earthquake_pipeline.gold.feature_store"
PREDICTIONS_TABLE = "earthquake_pipeline.gold.ml_predictions"

ML_FEATURES = [
    "latitude", "longitude", "abs_latitude",
    "depth_km", "depth_log",
    "sig", "nst", "gap", "rms", "dmin",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_shallow", "is_subduction_zone",
    "geo_region_enc", "depth_class_enc", "mag_type_enc"
]

print("    Configurações carregadas")
print(f"   Fonte   : {FEATURE_TABLE}")
print(f"   Destino : {PREDICTIONS_TABLE}")

# COMMAND ----------

# Carregar modelos do MLflow Registry para batch inference

MODEL_NAME_CLF = "earthquake-risk-classifier"
MODEL_NAME_REG = "earthquake-magnitude-regressor"

print("Carregando modelos do MLflow Registry...\n")

model_clf = mlflow.spark.load_model(f"models:/{MODEL_NAME_CLF}/latest")
model_reg = mlflow.spark.load_model(f"models:/{MODEL_NAME_REG}/latest")

print(f"Modelos carregados")
print(f"   Classificador : {MODEL_NAME_CLF}")
print(f"   Regressor     : {MODEL_NAME_REG}")

# COMMAND ----------

# DBTITLE 1,Rodar batch inference em todo o dataset
print("Rodando batch inference...\n")

df_all = spark.table(FEATURE_TABLE) \
              .dropna(subset=ML_FEATURES)

# Predições do classificador
pred_clf = model_clf.transform(df_all) \
    .withColumnRenamed("prediction", "pred_risk_level") \
    .select("event_id", "pred_risk_level")

# Predições do regressor
pred_reg = model_reg.transform(df_all) \
    .withColumnRenamed("prediction", "pred_magnitude") \
    .select("event_id", "pred_magnitude")

# Buscar APENAS place da Silver (tsunami_warning já existe na feature store)
df_silver_extra = spark.table("earthquake_pipeline.silver.earthquakes") \
    .select("event_id", "place") \
    .dropDuplicates(["event_id"])

# Juntar tudo
df_predictions = (
    df_all
    .join(pred_clf,        on="event_id", how="left")
    .join(pred_reg,        on="event_id", how="left")
    .join(df_silver_extra, on="event_id", how="left")

    .withColumn("pred_risk_label",
        F.when(F.col("pred_risk_level") == 3, "CRITICAL")
         .when(F.col("pred_risk_level") == 2, "HIGH")
         .when(F.col("pred_risk_level") == 1, "MEDIUM")
         .otherwise("LOW"))

    .withColumn("real_risk_label",
        F.when(F.col("risk_level_enc") == 3, "CRITICAL")
         .when(F.col("risk_level_enc") == 2, "HIGH")
         .when(F.col("risk_level_enc") == 1, "MEDIUM")
         .otherwise("LOW"))

    .withColumn("magnitude_error",
        F.round(F.abs(F.col("magnitude") - F.col("pred_magnitude")), 4))

    .withColumn("clf_correct",
        F.when(F.col("risk_level_enc") ==
               F.col("pred_risk_level"), True).otherwise(False))

    .withColumn("inference_at", F.current_timestamp())

    .select(
        "event_id", "event_time", "place",
        "magnitude", "pred_magnitude", "magnitude_error",
        "risk_level_enc", "pred_risk_level",
        "real_risk_label", "pred_risk_label", "clf_correct",
        "geo_region", "depth_km", "depth_class",
        "tsunami_warning", "sig",
        "latitude", "longitude",
        "inference_at"
    )
)

total = df_predictions.count()
print(f"Inferência concluída: {total:,} predições geradas")

# COMMAND ----------

# Salvar predições como Delta

print("Salvando tabela de predições...\n")

df_predictions.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(PREDICTIONS_TABLE)

print(f"Tabela salva: {PREDICTIONS_TABLE}")

# Estatísticas gerais
print("\nAcurácia por classe:")
display(spark.sql("""
    SELECT
        real_risk_label,
        COUNT(*)                                          AS total,
        SUM(CASE WHEN clf_correct THEN 1 ELSE 0 END)     AS corretos,
        ROUND(AVG(CASE WHEN clf_correct THEN 1.0 ELSE 0 END) * 100, 2)
                                                          AS accuracy_pct,
        ROUND(AVG(magnitude_error), 4)                    AS mae_regressao
    FROM earthquake_pipeline.gold.ml_predictions
    GROUP BY real_risk_label
    ORDER BY real_risk_label
"""))

print("\nExemplos de predições — eventos CRITICAL:")
display(spark.sql("""
    SELECT
        place, magnitude, pred_magnitude, magnitude_error,
        real_risk_label, pred_risk_label, clf_correct,
        geo_region, depth_km, tsunami_warning
    FROM earthquake_pipeline.gold.ml_predictions
    WHERE real_risk_label = 'CRITICAL'
    ORDER BY magnitude DESC
    LIMIT 10
"""))

# COMMAND ----------

# =============================================================================
# CÉLULA 5 — Relatório final do pipeline de ML
# =============================================================================
print("""

          EARTHQUAKE ML PIPELINE — RELATÓRIO FINAL            
                                                              
  DADOS
    Período    : Mar/2025 → Mar/2026 (13 meses)    
    Registros  : 28.700 eventos sísmicos           
    Features   : 19 features engenheiradas         
                                                   
  CLASSIFICAÇÃO (Risk Level — 4 classes)           
    Accuracy   : 98.90%                            
    F1 Score   : 0.9888                            
    Melhor clf : RandomForestClassifier (100 trees)
                                                   
  REGRESSÃO (Magnitude)                            
    R²         : 0.9881                            
    RMSE       : 0.0952                            
    Melhor reg : RandomForestRegressor (100 trees) 
                                                   
  MLFLOW REGISTRY                                  
    earthquake-risk-classifier        v1        
    earthquake-magnitude-regressor    v1       
                                                   
  TOP 3 FEATURES MAIS IMPORTANTES                  
    1. sig            (44.5%)                      
    2. mag_type_enc   (22.2%)                      
    3. geo_region_enc (11.8%)                      
                                                   

""")

print("PROJETO SEMANA 3 CONCLUÍDO!")
print("\n   Notebooks entregues:")
notebooks = [
    "00_data_collection.py   — Coleta histórica 13 meses USGS",
    "01_feature_engineering.py — 19 features + Feature Store Delta",
    "02_exploratory_analysis.py — EDA + correlações + qualidade",
    "03_model_training.py    — 5 modelos treinados no MLflow",
    "04_model_evaluation.py  — Avaliação + Feature Importance + Registry",
    "05_batch_inference.py   — Predições em batch na tabela Gold",
]
for nb in notebooks:
    print(f"    {nb}")
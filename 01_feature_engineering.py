# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import math

# COMMAND ----------

SILVER_TABLE   = "earthquake_pipeline.silver.earthquakes"
FEATURE_TABLE  = "earthquake_pipeline.gold.feature_store"

print("Configurações carregadas")
print(f"   Fonte   : {SILVER_TABLE}")
print(f"   Destino : {FEATURE_TABLE}")

# COMMAND ----------

df = spark.table(SILVER_TABLE)

print(f"Silver carregado: {df.count():,} registros")
print(f"\n Colunas disponíveis:")
for c in df.columns:
    print(f"   {c}")

# COMMAND ----------

# Criar features

print("Criando features...\n")

PI = math.pi

df_features = (
    df

    # Encoding cíclico da hora (captura padrão circular)
    # sin/cos evita que hora 23 e hora 0 pareçam distantes para o modelo
    .withColumn("hour_sin",
        F.round(F.sin(F.col("event_hour") * (2 * PI / 24)), 6))
    .withColumn("hour_cos",
        F.round(F.cos(F.col("event_hour") * (2 * PI / 24)), 6))

    # Feature 3-4: Encoding cíclico do mês
    .withColumn("month_sin",
        F.round(F.sin(F.col("event_month") * (2 * PI / 12)), 6))
    .withColumn("month_cos",
        F.round(F.cos(F.col("event_month") * (2 * PI / 12)), 6))

    # Flag de terremoto raso (maior risco de dano)
    .withColumn("is_shallow",
        F.when(F.col("depth_km") <= 70, 1).otherwise(0))

    # Flag de zona de subducção (maior risco de tsunami)
    .withColumn("is_subduction_zone",
        F.when(
            (F.col("geo_region").isin("Pacific / Other", "Asia", "Oceania")) &
            (F.col("depth_km") <= 70), 1
        ).otherwise(0))

    # Label encoding de geo_region
    .withColumn("geo_region_enc",
        F.when(F.col("geo_region") == "North America",     0)
         .when(F.col("geo_region") == "Pacific / Other",   1)
         .when(F.col("geo_region") == "Asia",              2)
         .when(F.col("geo_region") == "South America",     3)
         .when(F.col("geo_region") == "Oceania",           4)
         .when(F.col("geo_region") == "Europe",            5)
         .when(F.col("geo_region") == "Africa & Middle East", 6)
         .otherwise(7))

    # Label encoding de depth_class
    .withColumn("depth_class_enc",
        F.when(F.col("depth_class") == "Shallow",      0)
         .when(F.col("depth_class") == "Intermediate", 1)
         .when(F.col("depth_class") == "Deep",         2)
         .otherwise(0))

    # Label encoding de mag_type
    .withColumn("mag_type_enc",
        F.when(F.col("mag_type") == "ml",  0)
         .when(F.col("mag_type") == "md",  1)
         .when(F.col("mag_type") == "mb",  2)
         .when(F.col("mag_type") == "mw",  3)
         .when(F.col("mag_type") == "mww", 4)
         .when(F.col("mag_type") == "mwr", 5)
         .otherwise(6))

    # Distância ao equador (proxy de zona sísmica)
    .withColumn("abs_latitude",
        F.round(F.abs(F.col("latitude")), 4))

    # Profundidade normalizada (log)
    .withColumn("depth_log",
        F.round(F.log1p(F.col("depth_km")), 4))

    # sig normalizado (preencher nulos com 0)
    .withColumn("sig",
        F.coalesce(F.col("sig"), F.lit(0)))

    # nst normalizado
    .withColumn("nst",
        F.coalesce(F.col("nst"), F.lit(10)))

    # gap (preencher nulos com 180 — pior caso)
    .withColumn("gap",
        F.coalesce(F.col("gap"), F.lit(180.0)))

    # rms (preencher nulos com 0)
    .withColumn("rms",
        F.coalesce(F.col("rms"), F.lit(0.0)))

    # dmin (preencher nulos com 0)
    .withColumn("dmin",
        F.coalesce(F.col("dmin"), F.lit(0.0)))

    # risk_level_enc (para classificação)
    .withColumn("risk_level_enc",
        F.when(F.col("magnitude") >= 6.0, 3)   # CRITICAL
         .when(F.col("magnitude") >= 5.0, 2)   # HIGH
         .when(F.col("magnitude") >= 4.0, 1)   # MEDIUM
         .otherwise(0))                         # LOW

    # magnitude (para regressão) — já existe

    # Remover registros com features críticas nulas
    .filter(F.col("depth_km").isNotNull())
    .filter(F.col("latitude").isNotNull())
    .filter(F.col("longitude").isNotNull())
    .filter(F.col("magnitude").isNotNull())
)

total = df_features.count()
print(f"Features criadas: {total:,} registros")
print(f"   Colunas totais  : {len(df_features.columns)}")

# COMMAND ----------

df_features.select("risk_level_enc").distinct().display()

# COMMAND ----------

# Selecionar colunas finais da Feature Store

FEATURE_COLS = [
    # Identificação
    "event_id", "event_time",

    # Features geográficas
    "latitude", "longitude", "abs_latitude",
    "geo_region", "geo_region_enc",

    # Features de profundidade
    "depth_km", "depth_log", "depth_class", "depth_class_enc",
    "is_shallow", "is_subduction_zone",

    # Features temporais
    "event_hour", "event_month", "event_year",
    "hour_sin", "hour_cos", "month_sin", "month_cos",

    # Features sísmicas
    "sig", "nst", "gap", "rms", "dmin",
    "mag_type", "mag_type_enc",
    "net", "status",

    # Targets
    "magnitude",        # target regressão
    "magnitude_class",  # target classificação (texto)
    "risk_level_enc",   # target classificação (numérico)

    # Metadados
    "tsunami_warning",
    "_silver_processed_at"
]

df_final = df_features.select(FEATURE_COLS)

print(f"Feature store final: {df_final.count():,} registros")
print(f"   Features numéricas : {len(FEATURE_COLS)} colunas")

# COMMAND ----------

# Salvar Feature Store como Delta Lake

print("Salvando Feature Store...")

(
    df_final
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FEATURE_TABLE)
)

print(f"Feature Store salva: {FEATURE_TABLE}")

# Estatísticas das features numéricas
print("\n Estatísticas das features numéricas:")
display(spark.sql("""
    SELECT
        ROUND(AVG(magnitude),    3) AS avg_magnitude,
        ROUND(STDDEV(magnitude), 3) AS std_magnitude,
        ROUND(AVG(depth_km),     2) AS avg_depth_km,
        ROUND(STDDEV(depth_km),  2) AS std_depth_km,
        ROUND(AVG(sig),          2) AS avg_sig,
        ROUND(AVG(nst),          2) AS avg_nst,
        ROUND(AVG(gap),          2) AS avg_gap,
        ROUND(AVG(abs_latitude), 4) AS avg_abs_lat
    FROM earthquake_pipeline.gold.feature_store
"""))

print("\n Distribuição do target (risk_level_enc):")
display(spark.sql("""
    SELECT
        risk_level_enc,
        CASE risk_level_enc
            WHEN 3 THEN 'CRITICAL (M≥6.0)'
            WHEN 2 THEN 'HIGH (M5.0-5.9)'
            WHEN 1 THEN 'MEDIUM (M4.0-4.9)'
            ELSE        'LOW (M<4.0)'
        END AS risk_label,
        COUNT(*)                    AS total,
        ROUND(COUNT(*) * 100.0 /
            SUM(COUNT(*)) OVER (), 1) AS pct
    FROM earthquake_pipeline.gold.feature_store
    GROUP BY risk_level_enc
    ORDER BY risk_level_enc DESC
"""))

print("\n FEATURE ENGINEERING CONCLUÍDO!")
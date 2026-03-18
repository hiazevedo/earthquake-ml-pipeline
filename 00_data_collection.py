# Databricks notebook source
from datetime import datetime, timezone, timedelta
import requests
import json
import time
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType,
    DoubleType, IntegerType, ArrayType
)

# COMMAND ----------

# DBTITLE 1,Cell 2
# Reutilizamos o Volume do projeto
RAW_JSON_PATH = "/Volumes/earthquake_pipeline/bronze/raw_json"

# Meta: coletar dados dos últimos 365 dias em janelas de 30 dias
# A API USGS tem limite de 20.000 eventos por chamada
DIAS_HISTORICO  = 365
JANELA_DIAS     = 30
MIN_MAGNITUDE   = 2.5   # M≥2.5 garante eventos relevantes para ML

print("=" * 60)
print("  COLETA HISTÓRICA — earthquake-ml-pipeline")
print(f"  Meta: {DIAS_HISTORICO} dias de histórico | M≥{MIN_MAGNITUDE}")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Cell 3
# Função de coleta por janela de tempo
def fetch_window(start: datetime, end: datetime,
                 min_mag: float = 2.5) -> dict:
    """Coleta todos os eventos sísmicos entre start e end."""
    params = {
        "format":         "geojson",
        "starttime":      start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime":        end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude":   min_mag,
        "limit":          20000,
        "orderby":        "time"
    }
    resp = requests.get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params=params, timeout=60
    )
    resp.raise_for_status()
    return resp.json()


def save_window(data: dict, label: str) -> int:
    """Salva uma janela de dados como JSON no Volume."""
    eventos = len(data["features"])
    if eventos == 0:
        return 0

    filename = f"historical_{label}.json"
    filepath = f"{RAW_JSON_PATH}/{filename}"
    batch = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "total_events": eventos,
        "source":       "USGS Earthquake Hazards Program",
        "features":     data["features"]
    }
    dbutils.fs.put(filepath, json.dumps(batch), overwrite=True)
    return eventos

print("Funções definidas")

# COMMAND ----------

# DBTITLE 1,Cell 4
# Coleta histórica em janelas de 30 dias
print("Iniciando coleta histórica...\n")

end_global   = datetime.now(timezone.utc)
start_global = end_global - timedelta(days=DIAS_HISTORICO)

# Gerar janelas de 30 dias
janelas = []
cursor  = start_global
while cursor < end_global:
    window_end = min(cursor + timedelta(days=JANELA_DIAS), end_global)
    janelas.append((cursor, window_end))
    cursor = window_end

print(f"PERÍODO     : {start_global.strftime('%Y-%m-%d')} → {end_global.strftime('%Y-%m-%d')}")
print(f"JANELA      : {len(janelas)} janelas de {JANELA_DIAS} dias")
print(f"MAGNITUDE   : M≥{MIN_MAGNITUDE}\n")

total_eventos   = 0
total_arquivos  = 0
erros           = 0

for i, (w_start, w_end) in enumerate(janelas, 1):
    label = w_start.strftime("%Y%m")
    try:
        print(f"  [{i:02d}/{len(janelas)}] {w_start.strftime('%Y-%m-%d')} → "
              f"{w_end.strftime('%Y-%m-%d')} ...", end=" ")

        data    = fetch_window(w_start, w_end, MIN_MAGNITUDE)
        eventos = save_window(data, label)

        total_eventos  += eventos
        total_arquivos += 1
        print(f"OK {eventos:,} eventos")

        time.sleep(3)  # respeitar rate limit da API

    except Exception as e:
        erros += 1
        print(f"ERRO: {str(e)}")
        time.sleep(10)

print(f""" 
COLETA HISTÓRICA CONCLUÍDA
-   Janelas coletadas : {total_arquivos:<5}
-   Total de eventos  : {total_eventos:<7,}
-   Erros             : {erros:<5}

""")

# COMMAND ----------

# DBTITLE 1,Cell 5
# Reprocessar Bronze do zero com todos os 19 arquivos
print("Reprocessando Bronze com todos os 19 arquivos...\n")

geometry_schema = StructType([
    StructField("type",        StringType(), True),
    StructField("coordinates", ArrayType(DoubleType()), True)
])
properties_schema = StructType([
    StructField("mag",     DoubleType(),  True),
    StructField("place",   StringType(),  True),
    StructField("time",    LongType(),    True),
    StructField("updated", LongType(),    True),
    StructField("tz",      IntegerType(), True),
    StructField("url",     StringType(),  True),
    StructField("detail",  StringType(),  True),
    StructField("felt",    IntegerType(), True),
    StructField("cdi",     DoubleType(),  True),
    StructField("mmi",     DoubleType(),  True),
    StructField("alert",   StringType(),  True),
    StructField("status",  StringType(),  True),
    StructField("tsunami", IntegerType(), True),
    StructField("sig",     IntegerType(), True),
    StructField("net",     StringType(),  True),
    StructField("code",    StringType(),  True),
    StructField("magType", StringType(),  True),
    StructField("nst",     IntegerType(), True),
    StructField("dmin",    DoubleType(),  True),
    StructField("rms",     DoubleType(),  True),
    StructField("gap",     DoubleType(),  True),
    StructField("type",    StringType(),  True),
    StructField("title",   StringType(),  True),
])
feature_schema = StructType([
    StructField("type",       StringType(),      True),
    StructField("properties", properties_schema, True),
    StructField("geometry",   geometry_schema,   True),
    StructField("id",         StringType(),      True),
])
root_schema = StructType([
    StructField("collected_at", StringType(),              True),
    StructField("total_events", IntegerType(),             True),
    StructField("source",       StringType(),              True),
    StructField("features",     ArrayType(feature_schema), True),
])

RAW_JSON_PATH = "/Volumes/earthquake_pipeline/bronze/raw_json"
CHECKPOINT_B  = "/Volumes/earthquake_pipeline/checkpoints/streaming/bronze"

query_b = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", f"{CHECKPOINT_B}/schema")
    .option("multiLine", "true")
    .schema(root_schema)
    .load(RAW_JSON_PATH)
    .withColumn("feature",      F.explode("features"))
    .withColumn("event_id",     F.col("feature.id"))
    .withColumn("magnitude",    F.col("feature.properties.mag"))
    .withColumn("place",        F.col("feature.properties.place"))
    .withColumn("event_time",   F.to_timestamp(
                                  F.col("feature.properties.time") / 1000))
    .withColumn("updated_time", F.to_timestamp(
                                  F.col("feature.properties.updated") / 1000))
    .withColumn("alert",        F.col("feature.properties.alert"))
    .withColumn("status",       F.col("feature.properties.status"))
    .withColumn("tsunami",      F.col("feature.properties.tsunami"))
    .withColumn("sig",          F.col("feature.properties.sig"))
    .withColumn("net",          F.col("feature.properties.net"))
    .withColumn("mag_type",     F.col("feature.properties.magType"))
    .withColumn("felt",         F.col("feature.properties.felt"))
    .withColumn("nst",          F.col("feature.properties.nst"))
    .withColumn("dmin",         F.col("feature.properties.dmin"))
    .withColumn("rms",          F.col("feature.properties.rms"))
    .withColumn("gap",          F.col("feature.properties.gap"))
    .withColumn("longitude",    F.col("feature.geometry.coordinates")[0])
    .withColumn("latitude",     F.col("feature.geometry.coordinates")[1])
    .withColumn("depth_km",     F.col("feature.geometry.coordinates")[2])
    .withColumn("collected_at", F.to_timestamp("collected_at"))
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .drop("features", "feature")
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT_B}/writer")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable("earthquake_pipeline.bronze.earthquakes")
)
query_b.awaitTermination()

bronze_total = spark.table("earthquake_pipeline.bronze.earthquakes").count()
print(f"Bronze concluído: {bronze_total:,} registros")

# COMMAND ----------

# DBTITLE 1,Cell 6
# Reprocessar Silver do zero
print("Reprocessando Silver...\n")

CHECKPOINT_S = "/Volumes/earthquake_pipeline/checkpoints/streaming/silver"

query_s = (
    spark.readStream
    .format("delta")
    .table("earthquake_pipeline.bronze.earthquakes")
    .filter(F.col("event_id").isNotNull())
    .filter(F.col("magnitude").isNotNull())
    .withWatermark("event_time", "72 hours")
    .dropDuplicates(["event_id"])
    .withColumn("magnitude_class",
        F.when(F.col("magnitude") >= 7.0, "Major")
         .when(F.col("magnitude") >= 6.0, "Strong")
         .when(F.col("magnitude") >= 5.0, "Moderate")
         .when(F.col("magnitude") >= 4.0, "Light")
         .when(F.col("magnitude") >= 2.5, "Minor")
         .otherwise("Micro"))
    .withColumn("depth_class",
        F.when(F.col("depth_km") <= 70,  "Shallow")
         .when(F.col("depth_km") <= 300, "Intermediate")
         .otherwise("Deep"))
    .withColumn("tsunami_warning",
        F.when(F.col("tsunami") == 1, True).otherwise(False))
    .withColumn("geo_region",
        F.when(
            (F.col("latitude").between(-56, 15)) &
            (F.col("longitude").between(-82, -34)), "South America")
         .when(
            (F.col("latitude").between(15, 72)) &
            (F.col("longitude").between(-168, -52)), "North America")
         .when(
            (F.col("latitude").between(35, 72)) &
            (F.col("longitude").between(-25, 45)), "Europe")
         .when(
            (F.col("latitude").between(0, 55)) &
            (F.col("longitude").between(52, 150)), "Asia")
         .when(
            (F.col("latitude").between(-50, 0)) &
            (F.col("longitude").between(110, 180)), "Oceania")
         .otherwise("Pacific / Other"))
    .withColumn("event_year",  F.year("event_time"))
    .withColumn("event_month", F.month("event_time"))
    .withColumn("event_day",   F.dayofmonth("event_time"))
    .withColumn("event_hour",  F.hour("event_time"))
    .withColumn("magnitude",   F.round("magnitude", 2))
    .withColumn("depth_km",    F.round("depth_km",  2))
    .withColumn("_silver_processed_at", F.current_timestamp())
    .drop("alert", "tsunami", "_ingested_at")
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_S)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable("earthquake_pipeline.silver.earthquakes")
)
query_s.awaitTermination()

silver_total = spark.table("earthquake_pipeline.silver.earthquakes").count()
print(f"Silver concluído: {silver_total:,} registros")

print("\nDistribuição por magnitude_class:")
display(spark.sql("""
    SELECT magnitude_class,
           COUNT(*)                 AS total,
           ROUND(AVG(magnitude), 2) AS mag_media,
           ROUND(MIN(magnitude), 2) AS mag_min,
           ROUND(MAX(magnitude), 2) AS mag_max
    FROM earthquake_pipeline.silver.earthquakes
    GROUP BY magnitude_class
    ORDER BY mag_media DESC
"""))

print("\nDistribuição por ano/mês:")
display(spark.sql("""
    SELECT event_year, event_month,
           COUNT(*) AS total
    FROM earthquake_pipeline.silver.earthquakes
    GROUP BY event_year, event_month
    ORDER BY event_year, event_month
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC _________________________
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Caso precise reprocessar todo pipeline rodar as celulas abaixo primeiro
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 9
# Resetar checkpoints para forçar reprocessamento completo
print("RESETANDO checkpoints...")

checkpoints = [
    "/Volumes/earthquake_pipeline/checkpoints/streaming/bronze",
    "/Volumes/earthquake_pipeline/checkpoints/streaming/silver",
]

for cp in checkpoints:
    try:
        dbutils.fs.rm(cp, recurse=True)
        print(f"  OK Removido: {cp}")
    except:
        print(f"  AVISO  Não encontrado: {cp}")

print("\nOK Checkpoints resetados!")

# COMMAND ----------

# DBTITLE 1,Cell 10
# Limpar tabelas Bronze e Silver para reprocessar do zero
print("RESETANDO tabelas...")

spark.sql("DROP TABLE IF EXISTS earthquake_pipeline.bronze.earthquakes")
spark.sql("DROP TABLE IF EXISTS earthquake_pipeline.silver.earthquakes")

print("OK Tabelas removidas!")

# COMMAND ----------

# DBTITLE 1,Cell 11
# Verificar quantos arquivos temos no Volume
print("ARQUIVOS disponíveis no Volume:\n")

files = dbutils.fs.ls(RAW_JSON_PATH)
total_bytes = 0

for f in sorted(files, key=lambda x: x.name):
    total_bytes += f.size
    print(f"  ARQUIVO {f.name:<50} {f.size:>12,} bytes")

print(f"\n  Total: {len(files)} arquivos | {total_bytes/1024/1024:.1f} MB")
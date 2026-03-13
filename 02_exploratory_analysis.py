# Databricks notebook source
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

# COMMAND ----------

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor":  "#161b22",
    "axes.edgecolor":   "#30363d", "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor":  "#ffffff", "xtick.color":     "#8b949e",
    "ytick.color":      "#8b949e", "text.color":      "#c9d1d9",
    "grid.color":       "#21262d", "grid.linestyle":  "--",
    "grid.alpha":       0.5,       "font.family":     "monospace",
})
COLORS = ["#58a6ff","#3fb950","#ffa657","#f78166","#d2a8ff","#79c0ff","#56d364"]

df_pd = spark.table("earthquake_pipeline.gold.feature_store").toPandas()
print(f"Feature Store carregada: {len(df_pd):,} registros")
print(f"   Colunas: {len(df_pd.columns)}")

# COMMAND ----------


# Gráfico 1: Distribuição do target + features principais
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Distribuição de Magnitude
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df_pd["magnitude"], bins=40, color="#58a6ff",
         edgecolor="#0d1117", linewidth=0.4, alpha=0.85)
ax1.axvline(df_pd["magnitude"].mean(), color="#ffa657",
            linestyle="--", linewidth=2,
            label=f"Média: {df_pd['magnitude'].mean():.2f}")
ax1.set_title("Distribuição de Magnitude", fontweight="bold")
ax1.set_xlabel("Magnitude")
ax1.set_ylabel("Frequência")
ax1.legend(fontsize=8, framealpha=0.2)
ax1.grid(True, axis="y")

# Distribuição de Profundidade 
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df_pd["depth_km"], bins=50, color="#3fb950",
         edgecolor="#0d1117", linewidth=0.4, alpha=0.85)
ax2.axvline(70,  color="#ffa657", linestyle="--",
            linewidth=1.5, label="Shallow (70km)")
ax2.axvline(300, color="#f78166", linestyle="--",
            linewidth=1.5, label="Intermediate (300km)")
ax2.set_title("Distribuição de Profundidade", fontweight="bold")
ax2.set_xlabel("Profundidade (km)")
ax2.set_ylabel("Frequência")
ax2.legend(fontsize=8, framealpha=0.2)
ax2.grid(True, axis="y")

# Distribuição do Target
ax3 = fig.add_subplot(gs[0, 2])
labels = ["LOW\n(M<4)", "MEDIUM\n(M4-5)", "HIGH\n(M5-6)", "CRITICAL\n(M≥6)"]
counts = [df_pd[df_pd["risk_level_enc"]==i].shape[0] for i in range(4)]
bar_colors = ["#3fb950","#58a6ff","#ffa657","#f78166"]
bars = ax3.bar(labels, counts, color=bar_colors,
               edgecolor="#0d1117", linewidth=0.5)
for bar, val in zip(bars, counts):
    pct = val / len(df_pd) * 100
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 50,
             f"{val:,}\n({pct:.1f}%)",
             ha="center", va="bottom", fontsize=8, color="#ffffff")
ax3.set_title("Distribuição do Target (Risk Level)", fontweight="bold")
ax3.set_ylabel("Total de Eventos")
ax3.grid(True, axis="y")

# magnitude por região
ax4 = fig.add_subplot(gs[1, 0])
region_mag = df_pd.groupby("geo_region")["magnitude"].mean().sort_values(ascending=True)
ax4.barh(region_mag.index, region_mag.values,
         color="#d2a8ff", edgecolor="#0d1117", linewidth=0.4)
ax4.set_title("Magnitude Média por Região", fontweight="bold")
ax4.set_xlabel("Magnitude Média")
ax4.grid(True, axis="x")

# Depth by class
ax5 = fig.add_subplot(gs[1, 1])
depth_counts = df_pd["depth_class"].value_counts()
ax5.pie(depth_counts.values,
        labels=depth_counts.index,
        colors=COLORS[:3],
        autopct="%1.1f%%", startangle=90,
        textprops={"color":"#c9d1d9","fontsize":9},
        wedgeprops={"edgecolor":"#0d1117","linewidth":1.5})
ax5.set_title("Distribuição por Profundidade", fontweight="bold")

# Eventos por hora
ax6 = fig.add_subplot(gs[1, 2])
hour_counts = df_pd.groupby("event_hour").size()
ax6.fill_between(hour_counts.index, hour_counts.values,
                 alpha=0.3, color="#58a6ff")
ax6.plot(hour_counts.index, hour_counts.values,
         color="#58a6ff", linewidth=2, marker="o", markersize=3)
ax6.set_title("Eventos por Hora do Dia (UTC)", fontweight="bold")
ax6.set_xlabel("Hora")
ax6.set_ylabel("Total")
ax6.set_xticks(range(0, 24, 3))
ax6.grid(True)

plt.suptitle("Análise Exploratória — Earthquake Feature Store",
             fontsize=15, fontweight="bold", y=1.01, color="#ffffff")
plt.savefig("/tmp/eda_overview.png", bbox_inches="tight",
            facecolor="#0d1117", dpi=120)
plt.show()

# COMMAND ----------

# Gráfico 2: Matriz de correlação das features numéricas

NUMERIC_FEATURES = [
    "magnitude", "depth_km", "depth_log", "abs_latitude",
    "latitude", "longitude", "sig", "nst", "gap", "rms", "dmin",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_shallow", "is_subduction_zone", "risk_level_enc"
]

corr_matrix = df_pd[NUMERIC_FEATURES].corr()

fig, ax = plt.subplots(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, cmap="coolwarm",
    vmin=-1, vmax=1, center=0,
    annot=True, fmt=".2f", annot_kws={"size": 7},
    linewidths=0.3, linecolor="#0d1117",
    ax=ax, cbar_kws={"label": "Correlação"}
)
ax.set_title("Matriz de Correlação — Features Numéricas",
             fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.show()

# Top correlações com o target
print("Top correlações com 'magnitude' (target regressão):")
corr_target = corr_matrix["magnitude"].drop("magnitude") \
                                      .abs().sort_values(ascending=False)
display(corr_target.reset_index().rename(
    columns={"index":"feature","magnitude":"|correlacao|"}
).head(10))

# COMMAND ----------

# Gráfico 3: Magnitude vs features mais correlacionadas

TOP_FEATURES = ["sig", "depth_km", "nst", "gap", "abs_latitude"]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for i, feat in enumerate(TOP_FEATURES):
    sample = df_pd.sample(min(3000, len(df_pd)), random_state=42)
    scatter = axes[i].scatter(
        sample[feat], sample["magnitude"],
        c=sample["risk_level_enc"],
        cmap="RdYlGn_r", alpha=0.4, s=8,
        vmin=0, vmax=3
    )
    axes[i].set_title(f"magnitude vs {feat}", fontweight="bold", fontsize=9)
    axes[i].set_xlabel(feat, fontsize=8)
    axes[i].set_ylabel("magnitude" if i == 0 else "", fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes[-1], label="Risk Level")
plt.suptitle("Magnitude vs Top Features Correlacionadas",
             fontsize=13, fontweight="bold", color="#ffffff")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Gráfico 4: Boxplots de magnitude por região e profundidade

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Magnitude por região
region_order = df_pd.groupby("geo_region")["magnitude"] \
                    .median().sort_values(ascending=False).index
data_region  = [df_pd[df_pd["geo_region"] == r]["magnitude"].values
                for r in region_order]
bp1 = axes[0].boxplot(data_region, patch_artist=True,
                      medianprops={"color":"#ffa657","linewidth":2})
for i, patch in enumerate(bp1["boxes"]):
    patch.set_facecolor(COLORS[i % len(COLORS)])
    patch.set_alpha(0.7)
axes[0].set_xticklabels(region_order, rotation=25,
                        ha="right", fontsize=8)
axes[0].set_title("Distribuição de Magnitude por Região",
                  fontweight="bold")
axes[0].set_ylabel("Magnitude")
axes[0].grid(True, axis="y")

# Magnitude por profundidade
depth_order = ["Shallow", "Intermediate", "Deep"]
data_depth  = [df_pd[df_pd["depth_class"] == d]["magnitude"].values
               for d in depth_order]
bp2 = axes[1].boxplot(data_depth, patch_artist=True,
                      medianprops={"color":"#ffa657","linewidth":2})
depth_colors = ["#f78166","#58a6ff","#3fb950"]
for patch, color in zip(bp2["boxes"], depth_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_xticklabels(depth_order, fontsize=10)
axes[1].set_title("Distribuição de Magnitude por Profundidade",
                  fontweight="bold")
axes[1].set_ylabel("Magnitude")
axes[1].grid(True, axis="y")

plt.tight_layout()
plt.show()

# COMMAND ----------

# Relatório de qualidade das features para ML

print("=" * 60)
print("  RELATÓRIO DE QUALIDADE — FEATURES PARA ML")
print("=" * 60)

ML_FEATURES = [
    "latitude", "longitude", "abs_latitude", "depth_km", "depth_log",
    "sig", "nst", "gap", "rms", "dmin",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_shallow", "is_subduction_zone",
    "geo_region_enc", "depth_class_enc", "mag_type_enc"
]

print(f"\n{'Feature':<25} {'Nulos':>8} {'%Nulo':>7} "
      f"{'Média':>10} {'Std':>10} {'Min':>8} {'Max':>8}")
print("-" * 82)

for feat in ML_FEATURES:
    if feat in df_pd.columns:
        nulos = df_pd[feat].isna().sum()
        pct   = nulos / len(df_pd) * 100
        media = df_pd[feat].mean()
        std   = df_pd[feat].std()
        mn    = df_pd[feat].min()
        mx    = df_pd[feat].max()
        flag  = " X " if pct > 5 else ""
        print(f"{feat:<25} {nulos:>8,} {pct:>6.1f}% "
              f"{media:>10.3f} {std:>10.3f} {mn:>8.2f} {mx:>8.2f}{flag}")

print(f"\n EDA CONCLUÍDO!")
print(f"   Features prontas para treinamento: {len(ML_FEATURES)}")
print(f"   Registros disponíveis            : {len(df_pd):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlação: 
# MAGIC - sig tem correlação 0.94 com magnitude — é a feature mais preditiva. latitude (-0.47) e longitude (0.63) também são fortes.
# MAGIC - Classe desbalanceada: CRITICAL só tem 0.5% dos dados — vamos precisar de técnicas de balanceamento no treino.
# MAGIC - Profundidade: 80.1% dos terremotos são rasos (Shallow) — o modelo precisa aprender bem esse padrão.
# MAGIC - Qualidade: Apenas 18 nulos em depth_log (0.1%) — dataset praticamente limpo!
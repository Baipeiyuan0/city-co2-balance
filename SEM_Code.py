import os
import numpy as np
import pandas as pd
from semopy import Model
from semopy.stats import calc_stats
from sklearn.preprocessing import StandardScaler

file_path = r"REPLACE_WITH_YOUR_PATH\sample_data_SEM.xlsx"  # <-- edit this line
output_dir = os.path.dirname(os.path.abspath(file_path))
df = pd.read_excel(file_path)
df = df.dropna()

# Standardize numeric columns
cols_to_exclude = {'sin_FID', 'sou_FID'}
numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
cols_to_scale = sorted(list(numeric_cols - cols_to_exclude))
if cols_to_scale:
    scaler = StandardScaler()
    df.loc[:, cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# SEM model description
mod_desc = """
  H_S ~ Source_city_PC1 + Source_city_PC3
  H_L ~ Source_city_PC1 + Source_city_PC3
  H_L ~ Sink_city_PC1 + Sink_city_PC3
  H_S ~ Sink_city_PC1
  Sink_city_PC3 ~ H_S
  Source_city_PC3 ~ H_L
  Source_city_PC3 ~ H_S
"""
model = Model(mod_desc)
model.fit(df)

# Extract results
# Model parameters and estimates
results_df = model.inspect()
# Goodness-of-fit statistics
fit_stats_df = calc_stats(model)

os.makedirs(output_dir, exist_ok=True)
results_df.to_excel(os.path.join(output_dir, "sem_model_results.xlsx"), index=False)
fit_stats_df.to_excel(os.path.join(output_dir, "sem_model_fit_stats.xlsx"), index=False)


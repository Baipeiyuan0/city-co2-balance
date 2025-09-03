import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import shap
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Global figure style
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'axes.titlesize': 28,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black'
})
sns.set_style("whitegrid", {"grid.color": "gray", "grid.linestyle": "--", "grid.linewidth": 0.6})

# Data loading & preprocessing
file_path = r"F:\NATURE CITIES\投稿材料\代码和示例数据\sample_data_RF_meanSHAP.xlsx" # <-- edit this line
output_dir = os.path.dirname(file_path)
df = pd.read_excel(file_path)
min_year = df['year'].min()
df['Fac33'] = df['year'] - min_year
X = df.drop(columns=['FID', 'year', 'CR'])
y = df['CR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building & training
rf = RandomForestRegressor(
    n_estimators=1000,
    max_depth=35,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=15,
    bootstrap=True,
    random_state=42
)
rf.fit(X_train, y_train)
rf.feature_names_in_ = X.columns
for estimator in rf.estimators_:
    estimator.feature_names_in_ = X.columns

# Metrics (R2 and RMSE)
pred_train = rf.predict(X_train)
pred_test = rf.predict(X_test)
rmse_train = np.sqrt(np.mean((y_train - pred_train) ** 2))
rmse_test  = np.sqrt(np.mean((y_test - pred_test) ** 2))
r2_train = r2_score(y_train, pred_train)
r2_test  = r2_score(y_test, pred_test)
print(f"R² (train): {r2_train:.3f}")
print(f"R² (test): {r2_test:.3f}")
print(f"RMSE (train): {rmse_train:.3f}")
print(f"RMSE (test): {rmse_test:.3f}")

# Figure sizes and dpi
dpi = 300
figsize_std = (3508/dpi, 2480/dpi)
figsize_shap = (3508/dpi, 4960/dpi)
figsize_pdp = (7016/dpi, 4960/dpi)

# SHAP summary plot
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=figsize_shap, dpi=dpi)
shap.summary_plot(shap_values, X_test, show=False, cmap="coolwarm")
ax_shap = plt.gca()
ax_shap.set_xlabel("SHAP Value Impact", fontsize=20)
ax_shap.set_ylabel("Features", fontsize=20)
if plt.gcf().axes[-1]:
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=20)
    if cbar.get_ylabel():
        cbar.yaxis.label.set_size(20)
shap_summary_path = os.path.join(output_dir, "shap_summary_plot.tif")
plt.tight_layout()
plt.savefig(shap_summary_path, dpi=300, format="tiff")
plt.close()

# SHAP lollipop importance
avg_shap = np.mean(np.abs(shap_values), axis=0)
features = X_test.columns
order = np.argsort(avg_shap)[::-1]
sorted_features = features[order]
sorted_avg_shap = avg_shap[order]
plt.figure(figsize=figsize_std, dpi=dpi)
plt.hlines(y=range(len(sorted_features)), xmin=0, xmax=sorted_avg_shap,
           color="gray", alpha=0.7, linewidth=2)
colors = sns.color_palette("rocket", len(sorted_features))
plt.scatter(sorted_avg_shap, range(len(sorted_features)), color=colors,
            s=100, edgecolor="black", zorder=3)
plt.yticks(range(len(sorted_features)), sorted_features, fontsize=15)
plt.xlabel("Average |SHAP Value|", fontsize=20, labelpad=12)
plt.title("SHAP Values per Feature", fontsize=28, pad=20)
plt.xlim(0, sorted_avg_shap.max() * 1.05)
plt.gca().invert_yaxis()
plt.tight_layout()
shap_bar_path = os.path.join(output_dir, "shap_bar_plot.tif")
plt.savefig(shap_bar_path, dpi=300, format="tiff")
plt.close()

# Performance plots (train/test)
plot_data_train = pd.DataFrame({'Observed': y_train.values, 'Predicted': pred_train})
plt.figure(figsize=figsize_std, dpi=dpi)
sns.scatterplot(x='Observed', y='Predicted', data=plot_data_train,
                color='orange', s=80, edgecolor='black')
plt.plot([plot_data_train['Observed'].min(), plot_data_train['Observed'].max()],
         [plot_data_train['Observed'].min(), plot_data_train['Observed'].max()],
         color='gray', linestyle='--', linewidth=2)
sns.regplot(x='Observed', y='Predicted', data=plot_data_train,
            scatter=False, color='darkred', ci=95, line_kws={'linewidth': 2})
plt.title("Training Set: Observed vs. Predicted", fontsize=32, pad=12)
plt.xlabel("Observed CR", fontsize=26, labelpad=10)
plt.ylabel("Predicted CR", fontsize=26, labelpad=10)
handles_train = [
    Line2D([], [], marker='o', color='orange', markersize=8, linestyle='None', label='Training Data'),
    Line2D([], [], color='gray', linestyle='--', linewidth=2, label='1:1 Line'),
    Line2D([], [], color='darkred', linestyle='-', linewidth=2, label='Best Fit'),
    Patch(facecolor='mistyrose', edgecolor='none', alpha=0.7, label='95% CI')
]
plt.legend(handles=handles_train, loc="upper left", fontsize=18, frameon=False)
plt.text(0.75, 0.05, f"R² = {r2_train:.2f}\nRMSE = {rmse_train:.2f}",
         transform=plt.gca().transAxes, fontsize=18, verticalalignment='bottom')
train_perf_path = os.path.join(output_dir, "model_performance_train.tif")
plt.tight_layout()
plt.savefig(train_perf_path, dpi=300, format="tiff")
plt.close()

plot_data_test = pd.DataFrame({'Observed': y_test.values, 'Predicted': pred_test})
plt.figure(figsize=figsize_std, dpi=dpi)
sns.scatterplot(x='Observed', y='Predicted', data=plot_data_test,
                color='skyblue', s=80, edgecolor='black')
plt.plot([plot_data_test['Observed'].min(), plot_data_test['Observed'].max()],
         [plot_data_test['Observed'].min(), plot_data_test['Observed'].max()],
         color='gray', linestyle='--', linewidth=2)
sns.regplot(x='Observed', y='Predicted', data=plot_data_test,
            scatter=False, color='darkgreen', ci=95, line_kws={'linewidth': 2})
plt.title("Test Set: Observed vs. Predicted", fontsize=32, pad=12)
plt.xlabel("Observed CR", fontsize=26, labelpad=10)
plt.ylabel("Predicted CR", fontsize=26, labelpad=10)
handles_test = [
    Line2D([], [], marker='o', color='skyblue', markersize=8, linestyle='None', label='Test Data'),
    Line2D([], [], color='gray', linestyle='--', linewidth=2, label='1:1 Line'),
    Line2D([], [], color='darkgreen', linestyle='-', linewidth=2, label='Best Fit'),
    Patch(facecolor='palegreen', edgecolor='none', alpha=0.7, label='95% CI')
]
plt.legend(handles=handles_test, loc="upper left", fontsize=18, frameon=False)
plt.text(0.75, 0.05, f"R² = {r2_test:.2f}\nRMSE = {rmse_test:.2f}",
         transform=plt.gca().transAxes, fontsize=18, verticalalignment='bottom')
test_perf_path = os.path.join(output_dir, "model_performance_test.tif")
plt.tight_layout()
plt.savefig(test_perf_path, dpi=300, format="tiff")
plt.close()

# OOB error curve
oob_errors = []
n_estimators_list = list(range(50, 1001, 50))
rf_oob = RandomForestRegressor(
    n_estimators=50,
    warm_start=True,
    oob_score=True,
    max_depth=35,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=15,
    bootstrap=True,
    random_state=42
)
for n in n_estimators_list:
    rf_oob.n_estimators = n
    rf_oob.fit(X_train, y_train)
    if hasattr(rf_oob, "oob_prediction_"):
        oob_pred = rf_oob.oob_prediction_
        error = np.sqrt(np.mean((y_train - oob_pred) ** 2))
        oob_errors.append(error)
    else:
        oob_errors.append(np.nan)
plt.figure(figsize=figsize_std, dpi=dpi)
plt.plot(n_estimators_list, oob_errors, marker="o", linestyle="solid",
         color="darkorange", linewidth=2, label="OOB RMSE")
plt.xlabel("Number of Trees", fontsize=22, labelpad=12)
plt.ylabel("OOB RMSE", fontsize=22, labelpad=12)
plt.title("OOB Error Curve", fontsize=28, pad=20)
plt.grid(True, linestyle="--", linewidth=0.8, color="gray", alpha=0.7)
min_error = np.nanmin(oob_errors)
min_idx = n_estimators_list[np.argmin(oob_errors)]
plt.annotate(f"Min OOB RMSE: {min_error:.3f}\nat {min_idx} trees",
             xy=(min_idx, min_error), xytext=(min_idx+50, min_error+0.05),
             arrowprops=dict(facecolor='black', arrowstyle="->"),
             fontsize=18)
plt.legend(loc="upper right", fontsize=18, frameon=False)
oob_error_path = os.path.join(output_dir, "oob_error_curve.tif")
plt.tight_layout()
plt.savefig(oob_error_path, dpi=300, format="tiff")
plt.close()

# PDP for all features
all_features = X.columns.tolist()
num_features = len(all_features)
cols = 6
rows = int(np.ceil(num_features / cols))
fig, axes = plt.subplots(rows, cols, figsize=figsize_pdp, dpi=dpi)
axes = axes.flatten()
for i, feature in enumerate(all_features):
    grid = np.linspace(X[feature].min(), X[feature].max(), 20)
    pdp_mean = []
    for val in grid:
        X_temp = X.copy()
        X_temp[feature] = val
        tree_vals = [est.predict(X_temp).mean() for est in rf.estimators_]
        pdp_mean.append(np.mean(tree_vals))

    axes[i].plot(grid, pdp_mean, color="#f58a07", linewidth=2)
    axes[i].set_title(f"{feature}", fontsize=22, pad=10)
    axes[i].set_xlabel("Feature Value", fontsize=20, labelpad=8)
    axes[i].set_ylabel("")
    axes[i].tick_params(axis='both', labelsize=16)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
pdp_handle = Line2D([], [], color="#f58a07", linewidth=2, label="PDP")
fig.legend(handles=[pdp_handle], loc='lower center', ncol=1, fontsize=20)
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
pdp_plots_path = os.path.join(output_dir, "pdp_plots.tif")
plt.savefig(pdp_plots_path, dpi=300, format="tiff")
plt.close()

print("All figures saved to:", output_dir)


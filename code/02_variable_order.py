import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "turkiye_pisa_completed.csv")

features = ["PARED","BELONG","CULTPOS","HISEI"]
X = pd.DataFrame(StandardScaler().fit_transform(df[features]), columns=features)
y = df["reading"].values

# Exploratory standardized regression/ARIMAX-style coefficient ranking.
model = sm.tsa.SARIMAX(y, exog=X, order=(0,0,0), trend="c",
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

coef = pd.Series(model.params[1:1+len(features)], index=features)
se = pd.Series(model.bse[1:1+len(features)], index=features)
p = pd.Series(model.pvalues[1:1+len(features)], index=features)

vif = pd.Series(
    [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    index=features
)

res = pd.DataFrame({
    "variable": features,
    "standardized_beta": [coef[f] for f in features],
    "abs_beta": [abs(coef[f]) for f in features],
    "std_error": [se[f] for f in features],
    "p_value": [p[f] for f in features],
    "VIF": [vif[f] for f in features],
})
res = res.sort_values("abs_beta", ascending=False).reset_index(drop=True)
res["inclusion_order"] = range(1, len(res)+1)

out = ROOT / "results" / "variable_order_exploratory.csv"
res.to_csv(out, index=False)
print(res)

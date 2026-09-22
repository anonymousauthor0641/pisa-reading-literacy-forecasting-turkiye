import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
hist = pd.read_csv(ROOT / "data" / "processed" / "turkiye_pisa_completed.csv")
future = pd.read_csv(ROOT / "data" / "processed" / "future_exogenous_values.csv")

features = ["PARED","BELONG","CULTPOS","HISEI"]
orders = [(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,1,1)]

scaler = StandardScaler().fit(hist[features])
X = scaler.transform(hist[features])
XF = scaler.transform(future[features])

best = None
for order in orders:
    try:
        fit = SARIMAX(hist["reading"].values, exog=X, order=order, trend="c",
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        if np.isfinite(fit.aic) and (best is None or fit.aic < best[0]):
            best = (fit.aic, order, fit)
    except Exception:
        pass

if best is None:
    raise RuntimeError("No ARIMAX specification converged.")

aic, order, fit = best
pred = fit.get_forecast(steps=len(future), exog=XF)
ci = pred.conf_int(alpha=0.05)

out = pd.DataFrame({
    "year": future["year"],
    "forecast": pred.predicted_mean,
    "lower_95": ci[:,0],
    "upper_95": ci[:,1],
})
out.to_csv(ROOT / "results" / "final_ARIMAX_M4_forecast.csv", index=False)
print("Selected order:", order)
print("AIC:", aic)
print(out.to_string(index=False))

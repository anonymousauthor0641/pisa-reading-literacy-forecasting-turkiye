import math, random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "turkiye_pisa_completed.csv")

TEST_YEARS = [2015, 2018, 2022, 2025]
WINDOW = 3
SEEDS = [11, 23, 37, 51, 71]
ARIMA_ORDERS = [(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,1,1)]
MODEL_FEATURES = {
    "M0": [],
    "M1": ["PARED"],
    "M2": ["PARED","BELONG"],
    "M3": ["PARED","BELONG","CULTPOS"],
    "M4": ["PARED","BELONG","CULTPOS","HISEI"],
}

def dtw_distance(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    D = np.full((len(a)+1, len(b)+1), np.inf)
    D[0,0] = 0.0
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = abs(a[i-1]-b[j-1])
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[-1,-1])

def metrics(y, p):
    return {
        "MAE": mean_absolute_error(y,p),
        "MSE": mean_squared_error(y,p),
        "RMSE": math.sqrt(mean_squared_error(y,p)),
        "DTW": dtw_distance(y,p),
    }

def naive_predict(train):
    return float(train["reading"].iloc[-1])

def arimax_predict(train, testrow, feats):
    y = train["reading"].values
    ex_train = None
    ex_test = None
    scaler = None
    if feats:
        scaler = StandardScaler().fit(train[feats])
        ex_train = scaler.transform(train[feats])
        ex_test = scaler.transform(testrow[feats])
    best = None
    for order in ARIMA_ORDERS:
        try:
            m = SARIMAX(y, exog=ex_train, order=order, trend="c",
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            if np.isfinite(m.aic) and (best is None or m.aic < best[0]):
                best = (m.aic, order, m)
        except Exception:
            pass
    if best is None:
        return np.nan
    return float(best[2].forecast(1, exog=ex_test)[0])

def prophet_predict(train, testrow, feats):
    d = pd.DataFrame({"ds": pd.to_datetime(train["year"].astype(str) + "-01-01"),
                      "y": train["reading"].values})
    m = Prophet(growth="linear", yearly_seasonality=False,
                weekly_seasonality=False, daily_seasonality=False)
    scalers = {}
    for f in feats:
        sc = StandardScaler().fit(train[[f]])
        d[f] = sc.transform(train[[f]]).ravel()
        scalers[f] = sc
        m.add_regressor(f)
    m.fit(d)
    fut = pd.DataFrame({"ds":[pd.Timestamp(f"{int(testrow['year'].iloc[0])}-01-01")]})
    for f in feats:
        fut[f] = scalers[f].transform(testrow[[f]]).ravel()
    return float(m.predict(fut)["yhat"].iloc[0])

class RNNRegressor(nn.Module):
    def __init__(self, input_size, kind="LSTM", hidden=6):
        super().__init__()
        cls = nn.LSTM if kind=="LSTM" else nn.GRU
        self.rnn = cls(input_size, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:,-1,:]).squeeze(-1)

def make_windows(train, feats):
    cols = ["reading"] + feats
    vals = train[cols].values.astype(np.float32)
    X, y = [], []
    for i in range(len(vals)-WINDOW):
        X.append(vals[i:i+WINDOW])
        y.append(vals[i+WINDOW,0])
    return np.asarray(X), np.asarray(y)

def dl_predict(train, testrow, feats, kind, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cols = ["reading"] + feats
    scaler = StandardScaler().fit(train[cols])
    scaled = train.copy()
    scaled[cols] = scaler.transform(train[cols])

    X, y = make_windows(scaled, feats)
    if len(X) < 1:
        return np.nan

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = RNNRegressor(input_size=len(cols), kind=kind, hidden=6)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(100):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

    hist = pd.concat([train.tail(WINDOW), testrow.iloc[0:0]], ignore_index=True)
    x_raw = train[cols].tail(WINDOW).values.astype(np.float32)
    x_scaled = scaler.transform(pd.DataFrame(x_raw, columns=cols)).astype(np.float32)
    with torch.no_grad():
        pred_scaled = float(model(torch.tensor(x_scaled[None,:,:])).item())

    # invert only reading dimension
    dummy = np.zeros((1, len(cols)))
    dummy[0,0] = pred_scaled
    pred_raw = scaler.inverse_transform(dummy)[0,0]
    return float(pred_raw)

all_rows = []

for model_name, feats in MODEL_FEATURES.items():
    y_true = []
    preds = {"Naive":[], "ARIMA/ARIMAX":[], "Prophet":[]}
    dl_preds = {k:{s:[] for s in SEEDS} for k in ["LSTM","GRU"]}

    for year in TEST_YEARS:
        train = df[df["year"] < year].copy()
        test = df[df["year"] == year].copy()
        if len(test) != 1:
            continue
        y_true.append(float(test["reading"].iloc[0]))
        preds["Naive"].append(naive_predict(train))
        preds["ARIMA/ARIMAX"].append(arimax_predict(train, test, feats))
        preds["Prophet"].append(prophet_predict(train, test, feats))
        for kind in ["LSTM","GRU"]:
            for s in SEEDS:
                dl_preds[kind][s].append(dl_predict(train, test, feats, kind, s))

    for method in ["Naive","ARIMA/ARIMAX","Prophet"]:
        m = metrics(y_true, preds[method])
        all_rows.append({"model_set":model_name,"method":method,**m})

    for kind in ["LSTM","GRU"]:
        seed_metrics = [metrics(y_true, dl_preds[kind][s]) for s in SEEDS]
        row = {"model_set":model_name,"method":kind}
        for k in ["MAE","MSE","RMSE","DTW"]:
            vals = [sm[k] for sm in seed_metrics]
            row[k] = np.mean(vals)
            row[k+"_SD"] = np.std(vals, ddof=1)
        all_rows.append(row)

res = pd.DataFrame(all_rows)
out = ROOT / "results" / "backtesting_metrics_final.csv"
res.to_csv(out, index=False)
print(res.to_string(index=False))
print(f"\nSaved: {out}")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

TEST_YEARS = [2015, 2018, 2022, 2025]
WINDOW = 3
SEEDS = [11, 23, 37, 51, 71]

MODEL_FEATURES = {
    "M0": [],
    "M1": ["PARED"],
    "M2": ["PARED", "BELONG"],
    "M3": ["PARED", "BELONG", "CULTPOS"],
    "M4": ["PARED", "BELONG", "CULTPOS", "HISEI"],
}

ARIMA_ORDERS = [(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,1,1)]

DL_HIDDEN = 6
DL_EPOCHS = 100
DL_LR = 0.01
DL_WEIGHT_DECAY = 1e-4

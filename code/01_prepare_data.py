import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "data" / "raw" / "turkiye_pisa_reading_predictors.csv")

# Values used in the current manuscript analysis.
# BELONG 2003-2009: backward extrapolation.
# CULTPOS 2022-2025: forward extrapolation.
raw.loc[raw["year"].isin([2003,2006,2009]), "BELONG"] = [-0.132511,-0.134369,-0.136227]
raw.loc[raw["year"].isin([2022,2025]), "CULTPOS"] = [-0.640124,-0.774895]

out = ROOT / "data" / "processed" / "turkiye_pisa_completed.csv"
raw.to_csv(out, index=False)
print(out)
print(raw)

# Run order

Use Python 3.11+.

```bash
pip install -r requirements.txt
python code/01_prepare_data.py
python code/02_variable_order.py
python code/03_backtest_all_models.py
python code/04_final_arimax_m4.py
```

`03_backtest_all_models.py` is the decisive script for the final M0-M4 comparison.  
Please use its newly generated metrics in the manuscript tables before public release.

# Türkiye PISA Reading Literacy Forecasting

This repository contains the data preparation, model evaluation, and forecasting workflow used in the study:

**Data-Driven Forecasting of PISA Reading Literacy Performance of Türkiye: Artificial Intelligence Based Time Series Estimation**

## Models
- Naive Persistence
- ARIMA / ARIMAX
- Prophet
- LSTM
- GRU

## Model structures
- M0: Reading literacy only
- M1: M0 + PARED
- M2: M1 + BELONG
- M3: M2 + CULTPOS
- M4: M3 + HISEI

The predictor inclusion order was based on exploratory standardized ARIMAX coefficients together with VIF-based multicollinearity assessment.

## Validation
A common rolling-backtesting framework is used for the comparable model evaluation.  
For the recurrent models, the sliding-window length is **3 PISA observations**.

The common evaluation cycles used by the final code are:

**2015, 2018, 2022, and 2025**

For each test cycle, only observations available before that cycle are used for model fitting.  
After a test prediction is evaluated, the **observed value** is added to the historical series for the next backtesting step.

LSTM and GRU are evaluated over **5 independent random seeds**, and results are reported as mean ± standard deviation.

## Deep-learning configuration
- Hidden units: 6
- Recurrent layers: 1
- Epochs: 100
- Optimizer: Adam
- Learning rate: 0.01
- Weight decay: 1e-4
- Loss: MSE
- Training: full-batch
- Sliding-window size: 3

## ARIMA / ARIMAX
Candidate orders:
- (0,0,0)
- (1,0,0)
- (0,1,0)
- (1,1,0)
- (0,1,1)

The specification with the lowest AIC within each training fold is selected.

## Prophet
Linear growth is used and daily, weekly, and yearly seasonalities are disabled because the PISA observations are sparse assessment-cycle data rather than regularly sampled seasonal data.

## Metrics
- MAE
- MSE
- RMSE
- DTW

## Final forecasting
After model comparison, the final ARIMAX M4 model is fitted using all available observations through 2025 and projected exogenous inputs to obtain conditional forecasts for:
- 2028
- 2031
- 2034

These projections are exploratory and conditional. They should not be interpreted as deterministic future outcomes.

## Reproducibility note
PISA national means are treated as country-cycle observations. Sampling, plausible-value, and linking uncertainty are not propagated through the forecasting pipeline. Missing predictor observations are estimated using interpolation/extrapolation as documented in the manuscript and code.

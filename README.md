# Predicting Processing Time in Low-Resource Settings with Transformer Decoder-based Models


## Performance metrics of various models:

### Classical ML models

| Model                  | Params  | MAE   | MSE   | R^2   |
|------------------------|---------|-------|-------|-------|
| Ridge                  | 79      | 0.480 | 0.681 | 0.343 |
| Lasso                  | 79      | 0.518 | 0.746 | 0.280 |
| ElasticNet             | 79      | 0.489 | 0.696 | 0.328 |
| Regression Tree        | 6331    | **0.117** | 0.173 | 0.833 |
| **Random Forest Regression** | **100035** | **_0.111_** | **_0.109_** | **_0.895_** |

### LSTM-based models

| Model                  | Params  | MAE   | MSE   | R^2   |
|------------------------|---------|-------|-------|-------|
| LSTM                   | -       | 0.499 | 0.785 | -0.150|
| **BiLSTM**             | -       | 0.409 | **0.633** | **0.297** |
| TCN + LSTM             | -       | 0.500 | 0.765 | -0.039|
| TCN + BiLSTM           | -       | **0.398** | 0.651 | -0.162|

### Transformer Decoder-based models

| Model                        | Params  | MAE   | MSE   | R^2   |
|------------------------------|---------|-------|-------|-------|
| FFN                          | 320737  | 0.342 | 0.512 | 0.457 |
| **Transformer + FFN**        | **362529** | **0.245** | **0.246** | **0.706** |
| TCN + Transformer + FFN      | 375329  | 0.315 | 0.484 | 0.462 |


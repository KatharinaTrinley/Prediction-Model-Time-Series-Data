# Processing Time Prediction with TCN + biLSTM, TCN + Transformer and TCN + LSTM

We implement a hybrid TCN-biLSTM, TCN-LSTM and a TCN-Transformer architecture to improve the accuracy of shipment processing time predictions. By combining the efficient processing and temporal dependency capturing capabilities of TCNs with the self-attention mechanism of Transformers, or the sequential data handling strengths of LSTMs—specifically their ability to learn order dependencies in sequence prediction problems—this hybrid approach offers a robust solution for forecasting processing times in manufacturing contexts, among others:

### Figure A: TCN-biLSTM Architecture
![Kopie von Brainstorming (1)](https://github.com/KatharinaTrinley/Prediction-model-TCN-Transformer/assets/152901977/4f8e42f0-5677-48a1-b939-926030ae84e7)


### Figure B: TCN-LSTM Architecture
![Kopie von Brainstorming](https://github.com/KatharinaTrinley/Prediction-model-TCN-Transformer/assets/152901977/428675db-4883-46b4-881a-63de4c373253) 

### Figure C: TCN-Transformer Architecture 
![Brainstorming](https://github.com/KatharinaTrinley/Prediction-model-TCN-Transformer/assets/152901977/9e712f20-9827-4937-ba4e-633864136b50) 

## Comparative Analysis Results
Performance metrics of different models grouped by categories

| **Model**                               | **Params** | **MAE** | **MSE** |
|------------------------------------------|------------|---------|---------|
| **TCN + biLSTM**                         |            |         |         |
| TCN + biLSTM (50 units)                  | 101701     | **0.031**| **0.003**|
| TCN + biLSTM (25 units)                  | 45901      | 0.041   | 0.0042  |
| **TCN + LSTM**                           |            |         |         |
| TCN + LSTM (50 units)                    | 95193      | **0.055**| **0.006**|
| TCN + LSTM (25 units)                    | 63443      | 0.092   | 0.016   |
| **LSTM, biLSTM**                         |            |         |         |
| Long Short-Term Memory (50 units)        | 66251      | **0.050**| **0.007**|
| Long Short-Term Memory (25 units)        | 63443      | 0.097   | 0.022   |
| biLSTM (50 units)                        | 101701     | **0.028**| **0.002**|
| **TCN**                                  |            |         |         |
| Temporal Convolutional Network           | 52645      | 0.180   | 0.060   |
| **Transformer-related**                  |            |         |         |
| TCN + Transformer                        | 86117      | **0.215**| **0.087**|
| TCN + Transformer with Sparse Self-Attention | 81957 | 0.263   | 0.134   |
| TCN + LSTM + Transformer                 | 225637     | 0.076   | 0.034   |
| **Linear Regression**                    |            |         |         |
| Linear Regression                        | 95193      | 0.530   | 0.754   |



## Files

### `preprocessed.txt`
Contains the preprocessed data used for training the model:

Each line represents a data point, showing various stages of the shipment process and associated attributes. The preprocessing included:
- Turning the timestamps to Unix format and extracting date, time, and weekday components.
- Aggregating package dimensions and counts per order.
- Handling missing values.
- One-hot encoding categorical data.
- Removing outliers.
- Normalizing features using Min-Max scaling & Z-score normalization.

### `tcn_transformer.py`
Main file that includes the TCN-Transformer model:

- A TCN layer acts as the initial feature extractor, using dilated causal convolutions to deal with temporal dependencies.
- Stacked TCN layers process the data and extract hierarchical temporal features.
- An Attention Mechanism enhances the model's ability to focus on relevant features by computing attention weights.
- Multi-head attention mechanisms handle different parts of the input sequence.
- Extracted features are input to a Transformer Decoder, where self-attention layers and feed-forward neural networks learn relationships across input sequences.
- Finally, a dense layer produces the final forecasted shipment processing time.

### `tcn_transformer_sparse-self-attention.py`
File that includes the TCN-Transformer model with sparse self-attention.

### `TCN + LSTM.py`
Main file that includes the TCN-LSTM model:

- Instead of using the Transformer architecture, the extracted features from the TCN layer are input into an LSTM network.

### `TCN.py`
Implements only a TCN. Baseline Model I.

### `LSTM.py`
Implements an LSTM. Baseline Model II.

### `lin_reg.py`
Implements a linear regression model. Baseline Model III.


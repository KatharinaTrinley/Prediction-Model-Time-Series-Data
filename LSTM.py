import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

# Load and preprocess the data
data = pd.read_csv('/content/preprocessed.txt')
print("Original Dataset:")
print(data.head())

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
print("Normalized Dataset (Features):")
print(X_scaled[:5])  # Print first 5 rows of normalized features
print("Normalized Dataset (Target):")
print(y_scaled[:5])  # Print first 5 normalized target values
sequence_length = 10
X_reshaped = []
y_reshaped = []

for i in range(len(X_scaled) - sequence_length):
    X_reshaped.append(X_scaled[i:i + sequence_length])
    y_reshaped.append(y_scaled[i + sequence_length])

X_reshaped = np.array(X_reshaped)
y_reshaped = np.array(y_reshaped)

print("Reshaped Dataset (Features):")
print(X_reshaped[:2])  # Print first 2 sequences of features
print("Reshaped Dataset (Target):")
print(y_reshaped[:2])  # Print first 2 target values corresponding to the sequences

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)


# LSTM layers
lstm1 = LSTM(units=50, return_sequences=True)(input_layer)
dropout_lstm1 = Dropout(0.2)(lstm1)
lstm2 = LSTM(units=50)(dropout_lstm1)
dropout_lstm2 = Dropout(0.2)(lstm2)

# Fully connected layers
dense1 = Dense(50, activation='relu')(dropout_lstm2)
output_layer = Dense(1)(dense1)

# Create and compile the model
model = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

# Plot training history
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

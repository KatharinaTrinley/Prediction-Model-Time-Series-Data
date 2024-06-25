import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense, Input, Add
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/content/preprocessed.txt')

eingang_ts = ['date_EINGANGSDATUM_UHRZEIT', 'time_EINGANGSDATUM_UHRZEIT', 'weekday_EINGANGSDATUM_UHRZEIT']
verpackt_ts = ['date_VERPACKT_DATUM_UHRZEIT', 'time_VERPACKT_DATUM_UHRZEIT',
               'weekday_VERPACKT_DATUM_UHRZEIT', 'secs_VERPACKT_DATUM_UHRZEIT']
auftragsnummer_ts = ['date_AUFTRAGANNAHME_DATUM_UHRZEIT', 'time_AUFTRAGANNAHME_DATUM_UHRZEIT',
                     'weekday_AUFTRAGANNAHME_DATUM_UHRZEIT', 'secs_AUFTRAGANNAHME_DATUM_UHRZEIT']
lieferschein_ts = ['date_LIEFERSCHEIN_DATUM_UHRZEIT', 'time_LIEFERSCHEIN_DATUM_UHRZEIT',
                   'weekday_LIEFERSCHEIN_DATUM_UHRZEIT', 'secs_LIEFERSCHEIN_DATUM_UHRZEIT']
auftragannahme_ts = ['date_AUFTRAGANNAHME_DATUM_UHRZEIT', 'time_AUFTRAGANNAHME_DATUM_UHRZEIT',
                     'weekday_AUFTRAGANNAHME_DATUM_UHRZEIT', 'secs_AUFTRAGANNAHME_DATUM_UHRZEIT']
bereitgestellt_ts = ['date_BEREITGESTELLT_DATUM_UHRZEIT', 'time_BEREITGESTELLT_DATUM_UHRZEIT',
                     'weekday_BEREITGESTELLT_DATUM_UHRZEIT', 'secs_BEREITGESTELLT_DATUM_UHRZEIT']
TA_ts = ['weekday_TA_DATUM_UHRZEIT', 'date_TA_DATUM_UHRZEIT', 'time_TA_DATUM_UHRZEIT', 'secs_TA_DATUM_UHRZEIT']

# other data:
package_data = ['LAENGE_IN_CM', 'BREITE_IN_CM', 'HOEHE_IN_CM', 'GEWICHT_IN_KG', 'count_PACKSTUECKART=BEH',
                'count_PACKSTUECKART=CAR', 'count_PACKSTUECKART=GBP', 'count_PACKSTUECKART=PAL',
                'count_PACKSTUECKART=PKI', 'count_PACKSTUECKART=UNKNOWN', 'PACKAGE_COUNT']
auftragsnummer = ['category_AUFTRAGSNUMMER=DSGA', 'category_AUFTRAGSNUMMER=RBMANUSHIP', 'category_AUFTRAGSNUMMER=return']
land = ['LAND=AT', 'LAND=AUT', 'LAND=BE', 'LAND=BR', 'LAND=CH', 'LAND=CN', 'LAND=CZ', 'LAND=DE', 'LAND=DK', 'LAND=DR',
        'LAND=ES', 'LAND=FCA', 'LAND=FR', 'LAND=HU', 'LAND=IE', 'LAND=IN', 'LAND=IT', 'LAND=JP', 'LAND=KR', 'LAND=MX',
        'LAND=NL', 'LAND=None', 'LAND=PL', 'LAND=RO', 'LAND=RU', 'LAND=TR', 'LAND=UK', 'LAND=US']
sonderfahrt = ['SONDERFAHRT']
dienstleister = ['DIENSTLEISTER=DHL', 'DIENSTLEISTER=None', 'DIENSTLEISTER=TNT', 'DIENSTLEISTER=UPS']

step_1_features = eingang_ts + sonderfahrt
step_2_features = step_1_features + verpackt_ts + auftragsnummer_ts + package_data + auftragsnummer
step_3_features = step_2_features + land + auftragannahme_ts + auftragannahme_ts + lieferschein_ts
step_4_features = step_3_features + bereitgestellt_ts
step_5_features = step_4_features + TA_ts + dienstleister

# Function to augment data
def augment_data(data, num_augmentations=3):
    augmented_data = data.copy()
    for _ in range(num_augmentations):
        temp_data = data.copy()
        # Apply random noise
        temp_data['PROCESSING'] += np.random.normal(0, 0.01, size=temp_data['PROCESSING'].shape)
        augmented_data = pd.concat([augmented_data, temp_data], axis=0)
    return augmented_data

# Augment the data
data_augmented = augment_data(data, num_augmentations=3)

X = data_augmented[step_5_features]
y = data_augmented['PROCESSING']

# Replicate the data 4 times
X = np.tile(X, (4, 1))
y = np.tile(y, 4)

# Normalize the features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Reshape data for TCN
sequence_length = 10
X_reshaped = []
y_reshaped = []

for i in range(len(X_scaled) - sequence_length):
    X_reshaped.append(X_scaled[i:i + sequence_length])
    y_reshaped.append(y_scaled[i + sequence_length])

X_reshaped = np.array(X_reshaped)
y_reshaped = np.array(y_reshaped)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# Define a custom callback to print metrics at the end of each epoch
class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f} - mae: {logs['mae']:.4f} - val_mae: {logs['val_mae']:.4f} - mse: {logs['mse']:.4f} - val_mse: {logs['val_mse']:.4f}")

# Define the Transformer Decoder Layer
class TransformerDecoderLayer(Layer):
    def __init__(self, num_heads, dff, d_model, rate=0.3):
        super(TransformerDecoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = Sequential([Dense(dff, activation='relu'), Dense(d_model)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, look_ahead_mask=None):
        attn_output = self.mha(x, x, x, attention_mask=look_ahead_mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

# Build the TCN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, X_reshaped.shape[2])))
model.add(Dropout(0.3))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

# Add the Transformer decoder layer
decoder_layer = TransformerDecoderLayer(num_heads=1, dff=128, d_model=64, rate=0.3)
model.add(decoder_layer)

# Add final layers
model.add(Flatten())
model.add(Dense(50, activation='relu', kernel_regularizer='l2'))
model.add(Dense(1))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)

# Train the TCN model with transformer decoder
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[reduce_lr, MetricsCallback()])

# Evaluate the model performance
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense, Input, Add
from tensorflow.keras.layers import LayerNormalization, Layer, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention
 
def augment_data(data, num_augmentations=3):
    augmented_data = data.copy()
    for _ in range(num_augmentations):
        temp_data = data.copy()
        # Apply random noise
        temp_data['PROCESSING'] += np.random.normal(0, 0.01, size=temp_data['PROCESSING'].shape)
        augmented_data = pd.concat([augmented_data, temp_data], axis=0)
    return augmented_data

data_augmented = augment_data(data, num_augmentations=3)

X = data_augmented[step_5_features]
y = data_augmented['PROCESSING']

# Replicate the data 4 times
X = np.tile(X, (4, 1))
y = np.tile(y, 4)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

sequence_length = 10
X_reshaped = []
y_reshaped = []

for i in range(len(X_scaled) - sequence_length):
    X_reshaped.append(X_scaled[i:i + sequence_length])
    y_reshaped.append(y_scaled[i + sequence_length])

X_reshaped = np.array(X_reshaped)
y_reshaped = np.array(y_reshaped)

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# Metrics at end of Epoch
class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f} - mae: {logs['mae']:.4f} - val_mae: {logs['val_mae']:.4f} - mse: {logs['mse']:.4f} - val_mse: {logs['val_mse']:.4f}")

# Transformer Decoder Layer with Sparse Self-Attention Class
class TransformerDecoderLayer(Layer):
    def __init__(self, num_heads, dff, d_model, rate=0.3, sparsity=0.5):
        super(TransformerDecoderLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.sparsity = sparsity
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = Sequential([Dense(dff, activation='relu'), Dense(d_model)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def build(self, input_shape):
        self.query_dense = Dense(self.d_model)
        self.key_dense = Dense(self.d_model)
        self.value_dense = Dense(self.d_model)
        super(TransformerDecoderLayer, self).build(input_shape)

    def sparse_attention(self, scores):
        # Implement top-k sparsity
        k = int(self.sparsity * scores.shape[-1])
        top_k_values, _ = tf.math.top_k(scores, k=k, sorted=False)
        threshold = tf.reduce_min(top_k_values, axis=-1, keepdims=True)
        sparse_scores = tf.where(scores >= threshold, scores, tf.zeros_like(scores))
        return sparse_scores

    def call(self, x, training, look_ahead_mask=None):
        q = self.query_dense(x)
        k = self.key_dense(x)
        v = self.value_dense(x)

        attn_scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.d_model, tf.float32))
        if look_ahead_mask is not None:
            attn_scores += (look_ahead_mask * -1e9)

        sparse_attn_scores = self.sparse_attention(attn_scores)
        attn_weights = tf.nn.softmax(sparse_attn_scores, axis=-1)
        attn_output = tf.matmul(attn_weights, v)

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)
# TCN Model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, X_reshaped.shape[2])))
model.add(Dropout(0.3))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

# Transformer decoder layer with sparse self-attention
decoder_layer = TransformerDecoderLayer(num_heads=1, dff=128, d_model=64, rate=0.3, sparsity=0.5)
model.add(decoder_layer)

model.add(Flatten())
model.add(Dense(50, activation='relu', kernel_regularizer='l2'))
model.add(Dense(1))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)

# Train TCN model with transformer decoder
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[reduce_lr, MetricsCallback()])

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

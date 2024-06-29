from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Conv1D, Dropout, Dense, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam

data = pd.read_csv('/content/preprocessed.txt')
print("Original Dataset:")
print(data.head())
def augment_data(data, num_augmentations=3):
    augmented_data = data.copy()
    for _ in range(num_augmentations):
        temp_data = data.copy()
        # Apply random noise
        temp_data['PROCESSING'] += np.random.normal(0, 0.01, size=temp_data['PROCESSING'].shape)
        augmented_data = pd.concat([augmented_data, temp_data], axis=0)
    return augmented_data

data_augmented = augment_data(data, num_augmentations=3)
print("Augmented Dataset:")
print(data_augmented.head())

X = data_augmented[step_5_features]
y = data_augmented['PROCESSING']
X = np.tile(X, (4, 1))
y = np.tile(y, 4)
print("Replicated Dataset (Features):")
print(X[:5])  # Print first 5 rows of features
print("Replicated Dataset (Target):")
print(y[:5])  # Print first 5 target values


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

# TCN
input_layer = Input(shape=(sequence_length, X_train.shape[2]))
conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
dropout1 = Dropout(0.3)(conv1)
conv2 = Conv1D(filters=64, kernel_size=3, activation='relu')(dropout1)
dropout2 = Dropout(0.3)(conv2)
flatten = Flatten()(dropout2)

# LSTM
lstm1 = LSTM(units=25, return_sequences=True)(input_layer)
dropout_lstm1 = Dropout(0.2)(lstm1)
lstm2 = LSTM(units=25)(dropout_lstm1)
dropout_lstm2 = Dropout(0.2)(lstm2)

merged = Concatenate()([flatten, dropout_lstm2])
dense1 = Dense(50, activation='relu')(merged)
output_layer = Dense(1)(dense1)
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

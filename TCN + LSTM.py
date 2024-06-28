from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Conv1D, Dropout, Dense, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam

#TCN
input_layer = Input(shape=(sequence_length, X_train.shape[2]))
conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
dropout1 = Dropout(0.3)(conv1)
conv2 = Conv1D(filters=64, kernel_size=3, activation='relu')(dropout1)
dropout2 = Dropout(0.3)(conv2)
flatten = Flatten()(dropout2)

# LSTM 
lstm1 = LSTM(units=50, return_sequences=True)(input_layer)
dropout_lstm1 = Dropout(0.2)(lstm1)
lstm2 = LSTM(units=50)(dropout_lstm1)
dropout_lstm2 = Dropout(0.2)(lstm2)

merged = Concatenate()([flatten, dropout_lstm2])
dense1 = Dense(50, activation='relu')(merged)
output_layer = Dense(1)(dense1)
model = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

# Plot 
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

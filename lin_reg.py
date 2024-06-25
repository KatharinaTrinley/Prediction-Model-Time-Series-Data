from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# Flatten the data for linear regression
num_samples, seq_len, num_features = X_reshaped.shape
X_flattened = X_reshaped.reshape(num_samples, seq_len * num_features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y_reshaped, test_size=0.2, random_state=42)


model1 = LinearRegression()

model1.fit(X_train, y_train)
y_pred_test = model1.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
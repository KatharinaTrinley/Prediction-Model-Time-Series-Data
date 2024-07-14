import pandas as pd

data = pd.read_csv('preprocessed.csv')

priorities = set(data['SONDERFAHRT'].to_numpy())

print(priorities)
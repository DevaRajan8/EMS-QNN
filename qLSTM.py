import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/kaggle/input/datasetq/Dataset.csv')
print("Read File")

df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values('Time').reset_index(drop=True)

df.fillna(df.mean(numeric_only=True), inplace=True)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

numeric_cols = ['DHI', 'DNI', 'GHI', 'Wind_speed', 'Humidity', 'Temperature', 
                'PV_production', 'Wind_production', 'Electric_demand']
for col in numeric_cols:
    df = df[(df[col] >= lower_bound[col]) & (df[col] <= upper_bound[col])]

df = df.reset_index(drop=True)

input_features = ['Season', 'Day_of_the_week', 'DHI', 'DNI', 'GHI', 
                  'Wind_speed', 'Humidity', 'Temperature']
target_features = ['PV_production', 'Wind_production', 'Electric_demand']

scaler_input = MinMaxScaler(feature_range=(0, 2*np.pi))
scaler_target = MinMaxScaler()

df[input_features] = scaler_input.fit_transform(df[input_features])
df[target_features] = scaler_target.fit_transform(df[target_features])

window_size = 288  # 24 hours * 12 (5-minute intervals per hour)
prediction_steps = 12  # next hour (12 five-minute intervals)

X = []
y = []

for i in range(len(df) - window_size - prediction_steps + 1):
    X_window = df[input_features].iloc[i:i+window_size].values
    y_window = df[target_features].iloc[i+window_size:i+window_size+prediction_steps].values
    
    X.append(X_window)
    y.append(y_window)

X = np.array(X)
y = np.array(y)

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
y_train_reshaped = y_train.reshape(y_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
y_test_reshaped = y_test.reshape(y_test.shape[0], -1)
print("actual scaled data ranges:")
for col in input_features:
    print(f"{col}: {df[col].min():.2f} to {df[col].max():.2f}")

print(f"\nAll features now scaled to [0, {2*np.pi:.2f}] for quantum gates")

print(f"\nFirst few scaled values:")
print(df[input_features].head(3))

quantum_angles = df[input_features].values
print(f"\nQuantum angles array shape: {quantum_angles.shape}")

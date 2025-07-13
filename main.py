import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import matplotlib.pyplot as plt

df = pd.read_excel('C:/kev/code/servo_motor/dataset.xlsx')

X = df[['control_parameter_1', 'control_parameter_2']].values
Y = df[['angle', 'angular_velocity']].values

scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2)

def build_model():
    model = models.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
model = build_model()
model.summary()

history = model.fit(
    X_train, Y_train,
    epochs = 50,
    batch_size = 512,
    verbose = 1,
    validation_data = (X_test, Y_test)
)

# Plot loss curves
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/kev/code/servo_motor/loss_plot.png")
plt.show()

model.save("C:/kev/code/servo_motor/best_mlp_model.keras")
joblib.dump(scaler_X, 'C:/kev/code/servo_motor/scaler_X.save')
joblib.dump(scaler_Y, 'C:/kev/code/servo_motor/scaler_Y.save')


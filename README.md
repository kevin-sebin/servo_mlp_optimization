# Servo Motor Control Optimization using MLP

This project applies a Multi-Layer Perceptron (MLP) model to predict the angle and angular velocity of a servo motor based on two control parameters. After training, the model is used to perform gradient-based optimization to identify the control parameter combination that maximizes overall motor performance.

## Project Overview

- Input: Control Parameter 1 and Control Parameter 2  
- Output: Angle and Angular Velocity  
- Objective: Predict and maximize the sum of angle and angular velocity using learned model behavior.

The model is trained on a dataset of approximately 80,000 samples and optimized to generalize well beyond the given parameter groups.

## Directory Structure

servo_motor/
- dataset.xlsx # Input dataset
- main.py # Training script
- predict.py # Optimization script
- best_mlp_model.keras # Trained model
- scaler_X.save # Scaler for input features
- scaler_Y.save # Scaler for output targets
- loss_plot.png # Training/validation loss graph
- README.md # Project documentation

## Dataset Format

The dataset should be in Excel format with the following four columns:

- `control_parameter_1`  
- `control_parameter_2`  
- `angle`  
- `angular_velocity`

import tensorflow as tf
import numpy as np
import joblib

model = tf.keras.models.load_model("C:/kev/code/servo_motor/best_mlp_model.keras")
scaler_X = joblib.load("C:/kev/code/servo_motor/scaler_X.save")
scaler_Y = joblib.load("C:/kev/code/servo_motor/scaler_Y.save")

init_input = np.array([[0.5, 0.5]], dtype=np.float32)
input_var = tf.Variable(init_input)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch(input_var)
        output = model(input_var)
        score = tf.reduce_sum(output)
        loss = -score  

    grads = tape.gradient(loss, [input_var])
    optimizer.apply_gradients(zip(grads, [input_var]))
    input_var.assign(tf.clip_by_value(input_var, 0.0, 1.0))

    if step % 20 == 0:
        print(f"Step {step}: Current score = {score.numpy():.4f}")


optimal_input_real = scaler_X.inverse_transform(input_var.numpy())
optimal_output_scaled = model(input_var).numpy()
optimal_output_real = scaler_Y.inverse_transform(optimal_output_scaled)


print("\n========================")
print("Optimal Control Parameters:")
print(f"Control Param 1: {optimal_input_real[0, 0]:.4f}")
print(f"Control Param 2: {optimal_input_real[0, 1]:.4f}")
print("\n Predicted Outputs:")
print(f"Angle: {optimal_output_real[0, 0]:.4f}")
print(f"Angular Velocity: {optimal_output_real[0, 1]:.4f}")
print("========================\n")

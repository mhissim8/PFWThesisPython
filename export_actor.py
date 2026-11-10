import numpy as np
import tensorflow as tf
from pathlib import Path

folder = Path("results/ddpg")

# Load the trained DDPG actor
actor = tf.keras.models.load_model(
    folder / "actor.keras",
    compile=False
)

# Confirm the expected architecture and activations
layers = [
    layer for layer in actor.layers
    if isinstance(layer, tf.keras.layers.Dense)
]

assert len(layers) == 3, "Expected exactly 3 Dense layers"
assert [layer.units for layer in layers] == [128, 128, 8]
assert [layer.activation.__name__ for layer in layers] == [
    "relu", "relu", "tanh"
]

# Export the learned weights and biases
weights = {}

for i, layer in enumerate(layers):
    kernel, bias = layer.get_weights()
    weights[f"W{i}"] = kernel
    weights[f"b{i}"] = bias

np.savez(folder / "actor_weights.npz", **weights)

# Test the NumPy implementation against Keras
rng = np.random.default_rng(42)
states = rng.random((10, 12)).astype(np.float32)

keras_outputs = actor(states, training=False).numpy()

x = states.copy()

for i, layer in enumerate(layers):
    x = x @ weights[f"W{i}"] + weights[f"b{i}"]

    if i < 2:
        x = np.maximum(x, 0)
    else:
        x = np.tanh(x)

max_error = np.max(np.abs(keras_outputs - x))

print("Exported:", folder / "actor_weights.npz")
print("Input shape:", actor.input_shape)
print("Output shape:", actor.output_shape)
print("Maximum prediction difference:", max_error)

np.testing.assert_allclose(
    keras_outputs,
    x,
    rtol=1e-5,
    atol=1e-5
)

print("PASS: NumPy predictions match Keras.")

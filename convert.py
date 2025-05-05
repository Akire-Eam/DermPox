import tensorflow as tf

# Load your original model
model = tf.keras.models.load_model('efficienet.keras')

# Identify last convolutional layer
last_conv_layer = next(layer for layer in model.layers[::-1] 
                      if isinstance(layer, tf.keras.layers.Conv2D))

# Create model that outputs both predictions and conv layer activations
grad_model = tf.keras.models.Model(
    [model.inputs],
    [last_conv_layer.output, model.output]
)

# Convert to TFLite with both outputs
converter = tf.lite.TFLiteConverter.from_keras_model(grad_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model
with open('efficientnet_gradcam.tflite', 'wb') as f:
    f.write(tflite_model)
import tensorflow as tf
model = tf.keras.models.load_model('efficienet.keras')
model.save('efficienet.h5')
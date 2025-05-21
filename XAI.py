import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from math import ceil
from lime import lime_image

# Disable TensorFlow debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set paths
image_dir = "D:/Thesis/DermPox/Images"
output_dir = "D:/Thesis/DermPox/XAI"
os.makedirs(output_dir, exist_ok=True)

# Load trained model
model = tf.keras.models.load_model('final_deployment_model.keras')

# Get last conv layer for Grad-CAM
last_conv_layer = model.get_layer("convnext_base_stage_3_block_2_pointwise_conv_2")
grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

# Class labels
labels = {
    0: 'Chickenpox',
    1: 'Cowpox',
    2: 'Healthy',
    3: 'Measles',
    4: 'Monkeypox'
}

# Preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array, img

# Grad-CAM function
# def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
#     img_tensor = np.expand_dims(img_array, axis=0)

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_tensor)
#         if pred_index is None:
#             pred_index = tf.argmax(predictions[0])
#         class_channel = predictions[:, pred_index]

#     grads = tape.gradient(class_channel, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0)
#     heatmap /= tf.reduce_max(heatmap)
#     return heatmap.numpy()

# Collage generator
def create_collage(image_paths, collage_path, cols=3):
    images = [Image.open(p) for p in image_paths]
    img_width, img_height = images[0].size
    rows = ceil(len(images) / cols)
    collage_width = cols * img_width
    collage_height = rows * img_height
    collage_image = Image.new('RGB', (collage_width, collage_height), color=(255, 255, 255))

    for idx, img in enumerate(images):
        x = (idx % cols) * img_width
        y = (idx // cols) * img_height
        collage_image.paste(img, (x, y))

    collage_image.save(collage_path)
    print(f"Collage saved to {collage_path}")

# -----------------------------------------
# Step 1: Process and Save Grad-CAM images
# -----------------------------------------
# for category in os.listdir(image_dir):
#     category_path = os.path.join(image_dir, category)
#     if os.path.isdir(category_path):
#         image_file = os.listdir(category_path)[0]
#         image_path = os.path.join(category_path, image_file)

#         # Preprocess image
#         img_array, original_img = preprocess_image(image_path)

#         # Predict class
#         preds = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
#         pred_class = np.argmax(preds)

#         # Grad-CAM Heatmap
#         heatmap = make_gradcam_heatmap(img_array, grad_model, pred_class)
#         heatmap = cv2.resize(heatmap, (original_img.width, original_img.height))
#         heatmap = np.uint8(255 * heatmap)
#         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#         superimposed_img = cv2.addWeighted(
#             cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR),
#             0.6, heatmap, 0.4, 0
#         )

#         # Save Grad-CAM figure
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#         ax1.imshow(original_img)
#         ax1.set_title(f"(A) Original {labels[pred_class]}")
#         ax1.axis('off')
#         ax2.imshow(superimposed_img)
#         ax2.set_title("(B) Grad-CAM Heatmap")
#         ax2.axis('off')
#         plt.tight_layout()

#         gradcam_path = os.path.join(output_dir, f"{category}_GradCAM.png")
#         plt.savefig(gradcam_path, dpi=300, bbox_inches='tight')
#         plt.close()

#         print(f"Saved Grad-CAM for {category}: {gradcam_path}")

# Create Grad-CAM collage
# gradcam_images = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith("_GradCAM.png")]
# create_collage(gradcam_images, os.path.join(output_dir, "All_GradCAM_Collage.png"))

# -----------------------------------------
# Step 2: Process and Save LIME images
# -----------------------------------------
for category in os.listdir(image_dir):
    category_path = os.path.join(image_dir, category)
    if os.path.isdir(category_path):
        image_file = os.listdir(category_path)[0]
        image_path = os.path.join(category_path, image_file)

        # Preprocess image
        img_array, original_img = preprocess_image(image_path)

        # Predict class
        preds = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
        pred_class = np.argmax(preds)

        # LIME Explanation
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_array.astype(np.float32),
            model.predict,
            top_labels=1,
            hide_color=0,
            num_samples=500,
            batch_size=32
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )

        # Save LIME figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(original_img)
        ax1.set_title(f"(A) Original {labels[pred_class]}")
        ax1.axis('off')
        ax2.imshow(mask)
        ax2.set_title("(B) LIME Explanation")
        ax2.axis('off')
        plt.tight_layout()

        lime_path = os.path.join(output_dir, f"{category}_LIME.png")
        plt.savefig(lime_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved LIME for {category}: {lime_path}")

# Create LIME collage
lime_images = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith("_LIME.png")]
create_collage(lime_images, os.path.join(output_dir, "All_LIME_Collage.png"))

print("✅ All visualizations and collages completed!")
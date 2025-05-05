import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import lime
from lime import lime_image
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# Dark theme CSS with XAI section
st.markdown(f"""
<style>
    :root {{
        --primary: #2b5876;
        --secondary: #4e4376;
        --text: #e0e0e0;
        --background: #121212;
        --card-bg: #1e1e1e;
        --xai-bg: #252525;
    }}
    
    .stApp {{
        background-color: var(--background);
        color: var(--text);
    }}
    
    .header {{
        font-family: 'Arial', sans-serif;
        color: var(--text);
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }}
    
    .xai-container {{
        background: var(--xai-bg);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 2rem;
    }}
    
    .xai-header {{
        color: #4fc3f7;
        border-bottom: 1px solid #333;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }}
            
    .disclaimer-box {{
        background-color: #2d1e1e;  /* Dark red background */
        border-left: 4px solid #ff5252;
        border-radius: 4px;
        padding: 1rem;
        margin: 2rem 0;
    }}
    
    .disclaimer-title {{
        color: #ff5252;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }}
    
    .disclaimer-text {{
        color: #e0e0e0;
        font-size: 0.9rem;
    }}

    .prediction-card {{
        background-color: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }}
    .prediction-name {{
        font-size: 1.5rem;
        color: #4fc3f7;
        font-weight: bold;
        margin-bottom: 0.3rem;
    }}
    .confidence-bar-container {{
        width: 100%;
        background-color: #333;
        border-radius: 6px;
        height: 12px;
        overflow: hidden;
        margin: 0.5rem 0;
    }}
    .confidence-bar-fill {{
        height: 100%;
        background: linear-gradient(90deg, #4fc3f7, #2b5876);
    }}
    .secondary-card {{
        background-color: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
    }}
    .secondary-item {{
        margin-bottom: 1rem;
    }}
    .secondary-label {{
        color: #e0e0e0;
        font-weight: 500;
    }}
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="header">
    <h1 style="margin:0; font-size:2.2rem;">DermPox</h1>
    <p style="margin:0; font-size:1rem; color:#cfd8dc;">Skin Lesion Analysis with Explainable AI</p>
</div>
""", unsafe_allow_html=True)

# Load model with Grad-CAM support
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_model.keras')
    
    # Create model for Grad-CAM
    last_conv_layer = next(layer for layer in model.layers[::-1] 
                      if isinstance(layer, tf.keras.layers.Conv2D))
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [last_conv_layer.output, model.output]
    )
    return model, grad_model

model, grad_model = load_model()

# Classification labels
labels = {
    0: 'Chickenpox',
    1: 'Cowpox',
    2: 'Healthy',
    3: 'Measles',
    4: 'Monkeypox'
}

# Preprocessing - keep as uint8
# def preprocess_image(image):
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
#     image = image.resize((224, 224))
#     return np.array(image)  # Returns uint8 array (0-255)

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32)
    img = convnext_preprocess(img)  # [0, 1] normalization for ConvNeXt
    # img = efficientnet_preprocess(img)  
    return img

# Grad-CAM implementation (modified for uint8)
def make_gradcam_heatmap(img_array, grad_model):
    # Convert to float32 temporarily for calculations
    img_tensor = tf.convert_to_tensor(img_array.astype('float32'))
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    heatmap /= tf.math.reduce_max(heatmap) + 1e-8  # Normalize 0-1
    return heatmap.numpy()


# LIME explanation (works with uint8)
# def lime_explanation(img_array, model, top_labels=5):
#     explainer = lime_image.LimeImageExplainer()
#     explanation = explainer.explain_instance(
#         img_array,  # Already uint8
#         lambda x: model.predict(x.astype('uint8')),  # Ensure model gets uint8
#         top_labels=top_labels,
#         hide_color=0,
#         num_samples=700,
#         batch_size=32
#     )
#     return explanation

def lime_explanation(img_array, model, top_labels=5):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=img_array.astype('uint8'),
        classifier_fn=lambda x: model.predict(x.astype(np.float32)),  # Ensure float32 input for ConvNeXt
        top_labels=top_labels,
        hide_color=0,
        num_samples=700,  # reduced from 700
        batch_size=16
    )
    return explanation


# Main app
upload_option = st.radio(
    "Select input method:",
    ('Upload image', 'Use camera'),
    horizontal=True
)

uploaded_file = None
if upload_option == 'Use camera':
    uploaded_file = st.camera_input("Position the lesion in frame")
else:
    uploaded_file = st.file_uploader("Upload skin image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        img_array = preprocess_image(image)  # uint8 (0-255)
        img_tensor = np.expand_dims(img_array, axis=0)  # Keep as uint8
        
        st.markdown("### Image Preview")
        st.image(image, use_container_width=True)
        
        if st.button("Analyze Lesion"):
            with st.spinner("Analyzing..."):
                # Model receives uint8 input directly
                predictions = model.predict(img_tensor, verbose=0)[0]
                sorted_indices = np.argsort(predictions)[::-1]
                results = [(labels[i], float(predictions[i])) for i in sorted_indices]
                
                # Generate explanations
                heatmap = make_gradcam_heatmap(img_tensor, grad_model)
                explanation = lime_explanation(img_array, model)
            
            st.markdown("""
            <div class="disclaimer-box">
                <div class="disclaimer-title">⚠️ MEDICAL DISCLAIMER</div>
                <div class="disclaimer-text">
                    <p>This AI tool provides preliminary analysis only. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for accurate assessment.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display results
            st.markdown("## Analysis Results")
            
            # Top prediction
            # top_pred = results[0]
            # st.markdown(f"""
            # <div class="prediction-card">
            #     <div class="prediction-name">{top_pred[0]}</div>
            #     <div>Most likely diagnosis</div>
            #     <div class="confidence-meter">
            #         <div class="confidence-fill" style="width: {top_pred[1]*100:.1f}%"></div>
            #     </div>
            #     <div style="text-align: right; font-weight: 500; color: #4fc3f7">
            #         {top_pred[1]*100:.1f}% confidence
            #     </div>
            # </div>
            # """, unsafe_allow_html=True)

            top_pred = results[0]
            st.markdown(f"""
            <div class="prediction-card">
                <div style="font-size: 0.9rem; color: #cfd8dc; margin-bottom: 0.5rem;">PRIMARY DIAGNOSIS</div>
                <div class="prediction-name">{top_pred[0]}</div>
                <div style="font-size: 0.85rem; color: #cfd8dc;">Most probable condition</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar-fill" style="width: {top_pred[1]*100:.1f}%"></div>
                </div>
                <div style="text-align: right; color: #4fc3f7; font-weight: 500;">
                    {top_pred[1]*100:.1f}% confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Alternative diagnoses
            # Corrected alternative diagnoses section
            # with st.expander("View other possibilities"):
            #     for i in range(1, len(results)):  # Start from 1 to skip top prediction
            #         pred_label, pred_conf = results[i]
            #         st.markdown(f"""
            #         <div style="margin: 0.5rem 0; padding: 1rem; background: #333; border-radius: 8px;">
            #             <div style="font-weight: 500;">{pred_label}</div>
            #             <div class="confidence-meter">
            #                 <div class="confidence-fill" style="width: {pred_conf*100:.1f}%"></div>
            #             </div>
            #             <div style="text-align: right;">{pred_conf*100:.1f}%</div>
            #         </div>
            #         """, unsafe_allow_html=True)

            with st.expander("View other possibilities"):
            st.markdown(f"""
            <div class="secondary-card">
            """, unsafe_allow_html=True)
            for label, confidence in results[1:]:
                st.markdown(f"""
                <div class="secondary-item">
                    <div class="secondary-label">{label}</div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar-fill" style="width: {confidence*100:.1f}%"></div>
                    </div>
                    <div style="text-align: right; color: #4fc3f7;">{confidence*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Explainable AI Section
            st.markdown("""
            <div class="xai-container">
                <h3 class="xai-header">AI Explanation</h3>
                <p>See what features influenced the AI's decision:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Grad-CAM Visualization
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Grad-CAM Heatmap**")
                # heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
                # heatmap = np.uint8(255 * heatmap)
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
                heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # Make sure img_array is [0,1] float
                if img_array.max() > 1.0:
                    img_array = img_array / 255.0
                img_array = np.uint8(255 * img_array)

                # superimposed_img = cv2.addWeighted(img_array, 0.5, heatmap_color, 0.5, 0)
                superimposed_img = cv2.addWeighted(
                                    cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB),
                                    0.6,
                                    cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB),
                                    0.4,
                                    0
                                )
                st.image(superimposed_img, use_container_width=True)
            
            # LIME Visualization
            with col2:
                st.markdown("**LIME Explanation**")
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0],
                    positive_only=True,
                    num_features=5,
                    hide_rest=False
                )
                fig, ax = plt.subplots(figsize=(6,6))
                ax.imshow(mask)
                ax.axis('off')
                st.pyplot(fig, use_container_width=True)
            
            # Medical disclaimer
            st.markdown("""
            <div class="disclaimer">
                <p style="margin-bottom:0;">
                <strong>Note:</strong> This AI analysis is not a medical diagnosis. 
                Consult a healthcare professional for proper evaluation.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
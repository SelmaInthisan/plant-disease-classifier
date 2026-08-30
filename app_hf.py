"""
Hugging Face Spaces & Gradio Deployment Application for Plant Disease Classification.
1-Click Free Hosting on Hugging Face Spaces.
"""
import os
import io
import gradio as gr
from PIL import Image

from app.model_loader import ModelInferenceService
from app.disease_info import get_disease_info

# Initialize inference service
service = ModelInferenceService.get_instance()

def predict_leaf(image: Image.Image):
    if image is None:
        return (
            "Please upload or choose a leaf image.",
            {},
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        )

    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    result = service.predict(img_bytes)

    # Health status badge text
    is_healthy = result["is_healthy"]
    health_status = f"✅ HEALTHY ({result['confidence_percentage']})" if is_healthy else f"⚠️ DISEASED ({result['confidence_percentage']})"
    
    # Probabilities for Gradio Label component
    confidence_dict = result["class_probabilities"]

    primary_diagnosis = f"**Prediction:** {result['prediction']} ({result['confidence_percentage']} Confidence)\n\n**Latency:** {result['inference_time_ms']} ms"
    
    symptoms_text = "\n".join([f"• {s}" for s in result["symptoms"]])
    prevention_text = "\n".join([f"• {p}" for p in result["prevention"]])
    
    advisory_text = (
        f"**Pathogen Category:** {result['pathogen_type']}\n\n"
        f"**Observed Causes:** {result['causes']}\n\n"
        f"**Organic Remedy:**\n{result['organic_treatment']}\n\n"
        f"**Chemical Remedy:**\n{result['chemical_treatment']}"
    )

    return (
        health_status,
        confidence_dict,
        primary_diagnosis,
        symptoms_text,
        prevention_text,
        advisory_text
    )

sample_examples = [
    ["app/static/samples/sample_tomato_healthy.jpg"],
    ["app/static/samples/sample_potato_healthy.jpg"],
    ["app/static/samples/sample_pepper_healthy.jpg"],
    
    ["app/static/samples/sample_potato_early_blight.jpg"]
]
valid_examples = [ex for ex in sample_examples if os.path.exists(ex[0])]

with gr.Blocks(title="Plant Disease Classification — AI Leaf Health Diagnostics") as demo:
    gr.Markdown(
        """
        # 🌿 Plant Disease Classification System
        ### Deep-Learning-Powered Healthy vs. Diseased Leaf Diagnostics
        Upload a plant leaf photograph to classify it as Healthy or Diseased, view confidence probabilities, and receive general plant-health guidance.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Plant Leaf Image", sources=["upload", "webcam", "clipboard"])
            predict_btn = gr.Button("🔍 Analyze Plant Leaf", variant="primary", size="lg")
            
            if valid_examples:
                gr.Examples(
                    examples=valid_examples,
                    inputs=input_image,
                    label="Quick Test Samples"
                )

        with gr.Column(scale=1):
            health_output = gr.Textbox(label="Health Status", interactive=False)
            conf_output = gr.Label(num_top_classes=2, label="Class Probability Breakdown")
            primary_output = gr.Markdown(label="Diagnostic Summary")

    with gr.Row():
        with gr.Column():
            symptoms_output = gr.Textbox(label="🔍 Observable Symptoms", lines=4, interactive=False)
        with gr.Column():
            prevention_output = gr.Textbox(label="🛡️ Prevention & Cultural Practices", lines=4, interactive=False)

    with gr.Row():
        advisory_output = gr.Markdown(label="🌿 Plant Health Guidance")

    gr.Markdown(
        "**Important:** This model performs binary classification only (Healthy vs Diseased). "
        "It does not identify a specific disease or pathogen."
    )

    predict_btn.click(
        fn=predict_leaf,
        inputs=[input_image],
        outputs=[health_output, conf_output, primary_output, symptoms_output, prevention_output, advisory_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft(primary_hue="emerald"))

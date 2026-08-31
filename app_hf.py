"""
Hugging Face Spaces & Gradio Deployment Application
for Plant Disease Classification.
"""

import os
import gradio as gr
from PIL import Image

from app.model_loader import ModelInferenceService


# Initialize inference service
service = ModelInferenceService.get_instance()


def looks_like_leaf(image_path):
    """
    Conservative check for obviously non-leaf images.

    This is only an input filter. It does not modify or retrain
    the MobileNetV2 classification model.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((256, 256))

        pixels = list(image.getdata())

        if not pixels:
            return False

        total = len(pixels)

        # Estimate plant/green pixel proportion.
        green_pixels = sum(
            1
            for r, g, b in pixels
            if g > r * 1.05
            and g > b * 1.05
            and g > 45
        )

        green_ratio = green_pixels / total

        # Conservative threshold.
        # Many real leaves contain green, while obvious unrelated
        # images generally contain very little green.
        return green_ratio >= 0.08

    except Exception:
        # If validation cannot be performed, allow the image to
        # proceed to the classifier rather than blocking valid images.
        return True


def predict_leaf(image):
    """
    Analyze an uploaded plant leaf image.
    """

    if image is None:
        return (
            "Please upload or choose a leaf image.",
            {},
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        )

    # Reject obviously unrelated images.
    if not looks_like_leaf(image):
        return (
            "INVALID IMAGE",
            {},
            "Invalid Plant Leaf Image",
            "The uploaded image does not appear to contain a clear plant leaf.",
            "Please upload a clear photograph of a plant leaf.",
            "The system accepts plant leaf photographs for Healthy vs Diseased classification."
        )

    # Read uploaded image file.
    try:
        with open(image, "rb") as f:
            img_bytes = f.read()
    except Exception as e:
        return (
            f"Image Error: {str(e)}",
            {},
            "Unable to read image",
            "Please upload a valid image file.",
            "Try another clear plant leaf image.",
            "The image could not be processed."
        )

    # Run model prediction.
    try:
        result = service.predict(img_bytes)
    except Exception as e:
        return (
            f"Prediction Error: {str(e)}",
            {},
            "Prediction failed",
            "The model could not process this image.",
            "Please try another clear plant leaf image.",
            "Please check the application logs for more information."
        )

    # Health status.
    is_healthy = result["is_healthy"]

    if is_healthy:
        health_status = (
            f"HEALTHY ({result['confidence_percentage']})"
        )
    else:
        health_status = (
            f"DISEASED ({result['confidence_percentage']})"
        )

    # Class probabilities.
    confidence_dict = result["class_probabilities"]

    # Diagnostic summary.
    primary_diagnosis = (
        f"**Prediction:** {result['prediction']} "
        f"({result['confidence_percentage']} Confidence)\n\n"
        f"**Latency:** {result['inference_time_ms']} ms"
    )

    # Observable symptoms.
    symptoms = result.get("symptoms", [])

    if symptoms:
        symptoms_text = "\n".join(
            f"• {s}" for s in symptoms
        )
    else:
        symptoms_text = "No specific symptoms reported."

    # Prevention.
    prevention = result.get("prevention", [])

    if prevention:
        prevention_text = "\n".join(
            f"• {p}" for p in prevention
        )
    else:
        prevention_text = (
            "No specific prevention information available."
        )

    # Advisory information.
    advisory_text = (
        f"**Pathogen Category:** "
        f"{result.get('pathogen_type', 'N/A')}\n\n"

        f"**Observed Causes:** "
        f"{result.get('causes', 'N/A')}\n\n"

        f"**Organic Remedy:**\n"
        f"{result.get('organic_treatment', 'N/A')}\n\n"

        f"**Chemical Remedy:**\n"
        f"{result.get('chemical_treatment', 'N/A')}"
    )

    return (
        health_status,
        confidence_dict,
        primary_diagnosis,
        symptoms_text,
        prevention_text,
        advisory_text
    )# Six verified quick-test samples.
sample_examples = [
    ["app/static/samples/sample_tomato_healthy.jpg"],
    ["app/static/samples/sample_potato_healthy.jpg"],
    ["app/static/samples/sample_pepper_healthy.jpg"],
    
    ["app/static/samples/sample_potato_early_blight.jpg"]
]

valid_examples = [
    example
    for example in sample_examples
    if os.path.exists(example[0])
]


with gr.Blocks(
    title="Plant Disease Classification — AI Leaf Health Diagnostics"
) as demo:

    gr.Markdown(
        """
        # Plant Disease Classification System

        ### Deep-Learning-Powered Healthy vs. Diseased Leaf Diagnostics

        Upload a plant leaf photograph to classify its health status,
        view confidence probabilities, and obtain general agronomic
        advisory information.
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            input_image = gr.Image(
                type="filepath",
                label="Upload Plant Leaf Image",
                sources=[
                    "upload",
                    "webcam",
                    "clipboard"
                ]
            )

            predict_btn = gr.Button(
                "Analyze Plant Leaf",
                variant="primary",
                size="lg"
            )

            if valid_examples:
                gr.Examples(
                    examples=valid_examples,
                    inputs=input_image,
                    label="Quick Test Samples"
                )

        with gr.Column(scale=1):

            health_output = gr.Textbox(
                label="Health Status",
                interactive=False
            )

            conf_output = gr.Label(
                num_top_classes=2,
                label="Class Probability Breakdown"
            )

            primary_output = gr.Markdown(
                label="Diagnostic Summary"
            )

    with gr.Row():

        with gr.Column():

            symptoms_output = gr.Textbox(
                label="Observable Symptoms",
                lines=4,
                interactive=False
            )

        with gr.Column():

            prevention_output = gr.Textbox(
                label="Prevention & Cultural Practices",
                lines=4,
                interactive=False
            )

    with gr.Row():

        advisory_output = gr.Markdown(
            label="Organic & Chemical Treatment Recommendations"
        )

    predict_btn.click(
        fn=predict_leaf,
        inputs=[input_image],
        outputs=[
            health_output,
            conf_output,
            primary_output,
            symptoms_output,
            prevention_output,
            advisory_output
        ]
    )


if __name__ == "__main__":

    # Render supplies PORT through an environment variable.
    # Locally it defaults to 7860.
    port = int(
        os.environ.get("PORT", 7860)
    )

    demo.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False,
    show_error=True,
    max_file_size="10mb",
    theme=gr.themes.Soft(primary_hue="emerald")
)
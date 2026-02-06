import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import json
from pillow_heif import register_heif_opener
from tensorflow.keras.applications.efficientnet import preprocess_input
from utils.report import create_report

register_heif_opener()

stage1_model = tf.keras.models.load_model(
    "stage1_shrinkage_vs_structural_model_f.keras",
    custom_objects={"preprocess_input": preprocess_input}
)
stage2_model = tf.keras.models.load_model(
    "stage2_settlement_vs_vertical_model_f.keras",
    custom_objects={"preprocess_input": preprocess_input}
)

CLASSES_STAGE1 = ["shrinkage", "structural"]
CLASSES_STAGE2 = ["settlement", "vertical"]

with open("kb.json", "r") as f:
    KB = json.load(f)

def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict(img, surface_type):
    if img is None:
        return "Please upload an image", "", "", None

    arr = preprocess(img)

    pred1 = stage1_model.predict(arr)[0]
    idx1 = np.argmax(pred1)
    stage1_class = CLASSES_STAGE1[idx1]

    if stage1_class == "shrinkage":
        crack = "shrinkage"
    else:
        pred2 = stage2_model.predict(arr)[0]
        idx2 = np.argmax(pred2)
        crack = CLASSES_STAGE2[idx2]

    cause = KB[crack]["cause"]
    solution = KB[crack]["solution"]

    report_path = "crack_report.pdf"
    create_report(
        crack_type=crack,
        surface_type=surface_type,
        cause=cause,
        solution=solution,
        output_path=report_path
    )

    return (
        f"Crack Type: {crack}\nSurface: {surface_type}",
        cause,
        solution,
        report_path
    )

css = """
body {background-color: #0d0d0d;}
.gradio-container {background-color: #0d0d0d;}
label {color: white !important;}
"""

with gr.Blocks(css=css) as ui:

    gr.Markdown("<h1 style='text-align:center;'> Wall Crack Detection System</h1>")

    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload Crack Image")
        surface_input = gr.Dropdown(
            ["Wall", "Column", "Slab", "Beam Foundation"],
            label="Surface Type",
            value="Wall"
        )

    btn = gr.Button("Predict Crack")

    with gr.Row():
        out1 = gr.Textbox(
    label="Prediction",
    lines=3,
    max_lines=4
)
        out2 = gr.Textbox(label="Cause", lines=4)
        out3 = gr.Textbox(label="Recommended Solution", lines=4)

    file_out = gr.File(label="Download PDF Report")

    btn.click(
        predict,
        inputs=[img_input, surface_input],
        outputs=[out1, out2, out3, file_out]
    )

ui.launch(share=True)

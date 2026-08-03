
"""# Pneumonia Prediction Simulation"""

# ============================================================
# KWASU PNEUMONIA PROJECT - UI Test Version (No Tunnel yet)
# ============================================================


import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import datetime
import time

# -------------------- Load Models --------------------
import os
import gdown

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODELS = {
    "vgg19_model.h5": "1SE7YSTFUfmTonHdyLnRGnj30WvDIOfDg",
    "resnet50_model.h5": "1o3nzx2RucUVmWewKKfgbkj2Q8L0JYHng",
    "densenet121_model.h5": "1qwSEvcjSWhpslXgVHrA6cTwxdZ4CB8A2"
}

for filename, file_id in MODELS.items():
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            path,
            quiet=False
        )

vgg19 = tf.keras.models.load_model(os.path.join(MODEL_DIR, "vgg19_model.h5"))
resnet50 = tf.keras.models.load_model(os.path.join(MODEL_DIR, "resnet50_model.h5"))
densenet121 = tf.keras.models.load_model(os.path.join(MODEL_DIR, "densenet121_model.h5"))

print("✅ Models loaded successfully")


def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

def predict(img, selected_model):
    if img is None:
        empty = """
        <div style="height:300px; display:flex; align-items:center; justify-content:center;
                    background:#1e293b; border-radius:16px; border:1px solid #334155; color:#64748b;">
            Upload a Chest X-ray to begin
        </div>
        """
        return empty, "—", "—", "—", ""

    time.sleep(0.7)

    processed = preprocess(img)
    model_map = {
        "VGG19": vgg19,
        "ResNet50": resnet50,
        "DenseNet121": densenet121
    }

    results = {}
    if selected_model == "All Models (Recommended)":
        for name, model in model_map.items():
            pred = float(model.predict(processed, verbose=0)[0][0])
            label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
            conf = pred if pred > 0.5 else 1 - pred
            results[name] = (label, conf)
    else:
        pred = float(model_map[selected_model].predict(processed, verbose=0)[0][0])
        label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
        conf = pred if pred > 0.5 else 1 - pred
        results[selected_model] = (label, conf)
        for name in model_map:
            if name not in results:
                results[name] = ("—", 0.0)

    if selected_model == "All Models (Recommended)":
        votes = [results[n][0] for n in results]
        final_label = max(set(votes), key=votes.count)
        avg_conf = np.mean([results[n][1] for n in results])
    else:
        final_label = results[selected_model][0]
        avg_conf = results[selected_model][1]

    if final_label == "NORMAL":
        color = "#22c55e"
        glow = "rgba(34, 197, 94, 0.15)"
        status = "No Pneumonia Detected"
    else:
        color = "#ef4444"
        glow = "rgba(239, 68, 68, 0.15)"
        status = "Pneumonia Detected"

    result_html = f"""
    <div style="background:linear-gradient(160deg,#0f172a,#1e293b); border:1px solid #334155;
                border-radius:16px; padding:40px 30px; text-align:center;
                box-shadow:0 0 40px {glow}; min-height:300px;
                display:flex; flex-direction:column; justify-content:center;">
        <div style="font-size:12px; letter-spacing:2px; color:#94a3b8; margin-bottom:12px;">ANALYSIS RESULT</div>
        <div style="font-size:52px; font-weight:800; color:{color}; text-shadow:0 0 25px {color}; margin:8px 0;">
            {final_label}
        </div>
        <div style="font-size:16px; color:#cbd5e1; margin-bottom:30px;">{status}</div>
        <div style="font-size:12px; color:#94a3b8; letter-spacing:1px;">CONFIDENCE</div>
        <div style="font-size:34px; font-weight:700; color:white; margin:8px 0 18px 0;">{avg_conf*100:.1f}%</div>
        <div style="background:#0f172a; height:10px; width:75%; margin:0 auto; border-radius:50px; overflow:hidden;">
            <div style="background:{color}; height:100%; width:{avg_conf*100}%; border-radius:50px;"></div>
        </div>
    </div>
    """

    def card(name, label, conf):
        if conf == 0:
            return """<div style="background:#1e293b; border:1px solid #334155; border-radius:12px;
                                   height:90px; display:flex; align-items:center; justify-content:center; color:#475569;">—</div>"""
        c = "#22c55e" if label == "NORMAL" else "#ef4444"
        return f"""
        <div style="background:#1e293b; border:1px solid #334155; border-radius:12px; padding:14px; text-align:center; height:90px;">
            <div style="font-size:11px; color:#94a3b8; margin-bottom:4px;">{name}</div>
            <div style="font-size:16px; font-weight:700; color:{c};">{label}</div>
            <div style="font-size:13px; color:#e2e8f0;">{conf*100:.1f}%</div>
        </div>
        """

    vgg = card("VGG19", *results["VGG19"])
    res = card("ResNet50", *results["ResNet50"])
    den = card("DenseNet121", *results["DenseNet121"])

    details = f"""
    <div style="text-align:center; color:#94a3b8; font-size:13px; margin-top:12px;">
        {selected_model} &nbsp;•&nbsp; {datetime.datetime.now().strftime('%d %b %Y  %I:%M %p')}
    </div>
    """
    return result_html, vgg, res, den, details

# -------------------- CSS --------------------
css = """
.gradio-container {max-width: 1350px !important; margin: 0 auto !important; padding: 16px 24px !important;}
footer {display:none !important;}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="sky"), title="KWASU Pneumonia Project") as demo:

    # Header
    gr.HTML("""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:24px; padding-bottom:16px; border-bottom:1px solid #1e293b;">
        <div style="width:50px; height:50px; background:linear-gradient(135deg,#0ea5e9,#0284c7);
                    border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:24px;">
            🫁
        </div>
        <div>
            <div style="font-size:22px; font-weight:800; color:#f8fafc;">KWASU PNEUMONIA PROJECT</div>
            <div style="font-size:13px; color:#94a3b8;">VGG19 • ResNet50 • DenseNet121</div>
        </div>
    </div>
    """)

    # Stats
    with gr.Row():
        for title, value, sub in [
            ("Total Predictions", "1,248", "↑ 12.5%"),
            ("Pneumonia Detected", "532", "↑ 8.2%"),
            ("Normal Cases", "716", "↑ 15.3%"),
            ("Avg. Confidence", "86.4%", "↑ 4.7%"),
            ("Models", "3", "Active")
        ]:
            gr.HTML(f"""
            <div style="background:#1e293b; border:1px solid #334155; border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:12px; color:#94a3b8;">{title}</div>
                <div style="font-size:22px; font-weight:700; color:white; margin:4px 0;">{value}</div>
                <div style="font-size:12px; color:#4ade80;">{sub}</div>
            </div>
            """)

    gr.HTML("<div style='height:20px'></div>")

    # Main Area
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            gr.Markdown("### Make Prediction")
            input_img = gr.Image(type="pil", label="Upload Chest X-ray", height=280)
            model_choice = gr.Dropdown(
                choices=["All Models (Recommended)", "DenseNet121", "VGG19", "ResNet50"],
                value="All Models (Recommended)",
                label="Select Model"
            )
            btn = gr.Button("Predict Now", variant="primary", size="lg")

        with gr.Column(scale=6):
            gr.Markdown("### Result")
            result_out = gr.HTML()
            with gr.Row():
                vgg_out = gr.HTML()
                res_out = gr.HTML()
                den_out = gr.HTML()
            details_out = gr.HTML()

    gr.HTML("<div style='height:28px'></div>")

    # Model Performance
    gr.Markdown("### Model Performance")
    gr.HTML("""
    <div style="background:#1e293b; border:1px solid #334155; border-radius:16px; padding:28px 32px;">
        <div style="margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#e2e8f0; font-weight:600;">DenseNet121</span>
                <span style="color:#22c55e; font-weight:700;">86.70%</span>
            </div>
            <div style="background:#0f172a; height:8px; border-radius:50px; overflow:hidden;">
                <div style="background:#22c55e; width:86.7%; height:100%;"></div>
            </div>
        </div>
        <div style="margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#e2e8f0; font-weight:600;">VGG19</span>
                <span style="color:#38bdf8; font-weight:700;">86.54%</span>
            </div>
            <div style="background:#0f172a; height:8px; border-radius:50px; overflow:hidden;">
                <div style="background:#38bdf8; width:86.54%; height:100%;"></div>
            </div>
        </div>
        <div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#e2e8f0; font-weight:600;">ResNet50</span>
                <span style="color:#a78bfa; font-weight:700;">62.50%</span>
            </div>
            <div style="background:#0f172a; height:8px; border-radius:50px; overflow:hidden;">
                <div style="background:#a78bfa; width:62.5%; height:100%;"></div>
            </div>
        </div>
    </div>
    """)

    btn.click(
        fn=predict,
        inputs=[input_img, model_choice],
        outputs=[result_out, vgg_out, res_out, den_out, details_out],
        show_progress="full"
    )

import os

port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
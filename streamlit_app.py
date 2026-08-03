import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import gdown
import os

from PIL import Image
from tensorflow.keras.preprocessing import image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KWASU Pneumonia AI",
    page_icon="🫁",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main{
    background:#0f172a;
}

.block-container{
    padding-top:1rem;
}

.metric-card{
    background:#1e293b;
    padding:18px;
    border-radius:15px;
    border:1px solid #334155;
    text-align:center;
}

.metric-title{
    color:#94a3b8;
    font-size:14px;
}

.metric-value{
    color:white;
    font-size:28px;
    font-weight:bold;
}

.metric-sub{
    color:#22c55e;
    font-size:13px;
}

.result-box{

    background:#111827;
    border-radius:18px;
    padding:40px;
    text-align:center;
    border:1px solid #334155;

}

.sidebar .sidebar-content{
    background:#111827;
}

</style>
""",unsafe_allow_html=True)

# ============================================================
# DOWNLOAD MODELS
# ============================================================

MODEL_DIR="models"

os.makedirs(MODEL_DIR,exist_ok=True)

MODELS={

"vgg19_model.h5":"1SE7YSTFUfmTonHdyLnRGnj30WvDIOfDg",

"resnet50_model.h5":"1o3nzx2RucUVmWewKKfgbkj2Q8L0JYHng",

"densenet121_model.h5":"1qwSEvcjSWhpslXgVHrA6cTwxdZ4CB8A2"

}

for filename,fileid in MODELS.items():

    path=os.path.join(MODEL_DIR,filename)

    if not os.path.exists(path):

        with st.spinner(f"Downloading {filename}..."):

            gdown.download(
                f"https://drive.google.com/uc?id={fileid}",
                path,
                quiet=False
            )

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource

def load_models():

    vgg=tf.keras.models.load_model(
        os.path.join(MODEL_DIR,"vgg19_model.h5")
    )

    res=tf.keras.models.load_model(
        os.path.join(MODEL_DIR,"resnet50_model.h5")
    )

    dense=tf.keras.models.load_model(
        os.path.join(MODEL_DIR,"densenet121_model.h5")
    )

    return vgg,res,dense

vgg19,resnet50,densenet121=load_models()

# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess(img):

    img=img.convert("RGB")

    img=img.resize((224,224))

    arr=image.img_to_array(img)

    arr=np.expand_dims(arr,axis=0)

    arr=arr/255.0

    return arr

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(img,selected_model):

    processed=preprocess(img)

    model_map={

        "VGG19":vgg19,

        "ResNet50":resnet50,

        "DenseNet121":densenet121

    }

    results={}

    if selected_model=="All Models":

        for name,model in model_map.items():

            pred=float(model.predict(processed,verbose=0)[0][0])

            label="PNEUMONIA" if pred>0.5 else "NORMAL"

            conf=pred if pred>0.5 else 1-pred

            results[name]=(label,conf)

    else:

        pred=float(model_map[selected_model].predict(processed,verbose=0)[0][0])

        label="PNEUMONIA" if pred>0.5 else "NORMAL"

        conf=pred if pred>0.5 else 1-pred

        results[selected_model]=(label,conf)

    return results


    # ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🫁 Pneumonia AI")

st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox(

    "Choose Prediction Model",

    [

        "All Models",

        "DenseNet121",

        "VGG19",

        "ResNet50"

    ]

)

st.sidebar.markdown("---")

st.sidebar.info("""

### About

This application compares three deep learning models for automatic Pneumonia Detection from Chest X-ray images.

Models:

• DenseNet121

• VGG19

• ResNet50

""")

# ============================================================
# HEADER
# ============================================================

st.title("🫁 KWASU Pneumonia Detection System")

st.caption(
"Deep Learning-Based Pneumonia Detection using Transfer Learning"
)

st.markdown("---")

# ============================================================
# DASHBOARD CARDS
# ============================================================

c1,c2,c3,c4 = st.columns(4)

with c1:

    st.markdown("""

<div class="metric-card">

<div class="metric-title">

Models

</div>

<div class="metric-value">

3

</div>

<div class="metric-sub">

Available

</div>

</div>

""",unsafe_allow_html=True)

with c2:

    st.markdown("""

<div class="metric-card">

<div class="metric-title">

Input Size

</div>

<div class="metric-value">

224×224

</div>

<div class="metric-sub">

Pixels

</div>

</div>

""",unsafe_allow_html=True)

with c3:

    st.markdown("""

<div class="metric-card">

<div class="metric-title">

Framework

</div>

<div class="metric-value">

TensorFlow

</div>

<div class="metric-sub">

Keras

</div>

</div>

""",unsafe_allow_html=True)

with c4:

    st.markdown("""

<div class="metric-card">

<div class="metric-title">

Institution

</div>

<div class="metric-value">

KWASU

</div>

<div class="metric-sub">

Computer Science

</div>

</div>

""",unsafe_allow_html=True)

st.write("")

# ============================================================
# MAIN LAYOUT
# ============================================================

left,right = st.columns([1,1.2])

with left:

    uploaded = st.file_uploader(

        "Upload Chest X-ray",

        type=["png","jpg","jpeg"]

    )

    if uploaded:

        img = Image.open(uploaded)

        st.image(

            img,

            use_container_width=True

        )

with right:

    st.subheader("Prediction Result")

    if uploaded:

        with st.spinner("Running AI Analysis..."):

            results = predict(

                img,

                selected_model

            )

        if selected_model=="All Models":

            labels=[]

            confs=[]

            for m in results:

                labels.append(results[m][0])

                confs.append(results[m][1])

            final=max(

                set(labels),

                key=labels.count

            )

            confidence=np.mean(confs)

        else:

            final=results[selected_model][0]

            confidence=results[selected_model][1]

        if final=="NORMAL":

            st.success(

                f"Prediction: {final}"

            )

        else:

            st.error(

                f"Prediction: {final}"

            )

        st.progress(float(confidence))

        st.metric(

            "Confidence",

            f"{confidence*100:.2f}%"

        )

        st.write("")

        st.subheader("Individual Model Results")

        table=[]

        for name in results:

            table.append({

                "Model":name,

                "Prediction":results[name][0],

                "Confidence (%)":

                round(results[name][1]*100,2)

            })

        st.dataframe(

            pd.DataFrame(table),

            use_container_width=True,

            hide_index=True

        )


        # ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")
st.subheader("📊 Model Performance")

performance = pd.DataFrame({
    "Model": ["DenseNet121", "VGG19", "ResNet50"],
    "Accuracy": [86.70, 86.54, 62.50]
})

st.bar_chart(
    performance.set_index("Model")
)

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown("---")

st.subheader("📘 Project Information")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
**Project Title**

Deep Learning-Based Pneumonia Detection using Transfer Learning

**Institution**

Kwara State University (KWASU)

**Department**

Computer Science
""")

with col2:

    st.markdown("""
**Deep Learning Models**

- DenseNet121
- VGG19
- ResNet50

**Framework**

TensorFlow / Keras

**Deployment**

Streamlit
""")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
"""
© 2026 KWASU Final Year Project

Developed for academic purposes.
"""
)
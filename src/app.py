# app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from models.cnn import SimpleCNN
import scipy.ndimage as nd
import cv2

st.set_page_config(
    page_title="VisuNet",
    layout="centered"
)

st.title("VisuNet – Handwritten Digit Inference")

# ----------------------------
# Model loading
# ----------------------------


@st.cache_resource
def load_model():
    model = SimpleCNN()
    state = torch.load("data/mnist_cnn.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


model = load_model()

# ----------------------------
# Canvas
# ----------------------------
st.write("Draw a digit (0–9)")

canvas = st_canvas(
    fill_color="white",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ----------------------------
# Preprocess
# ----------------------------


def preprocess(canvas_img):
    # 1. RGBA → Gray
    img = cv2.cvtColor(canvas_img, cv2.COLOR_RGBA2GRAY)

    # 3. normalize [0,1]
    img = img / 255.0

    # 4. bbox of digit
    ys, xs = np.where(img > 0.05)
    if len(xs) == 0:
        return None, None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    digit = img[y_min:y_max+1, x_min:x_max+1]

    # 5. make square
    h, w = digit.shape
    size = max(h, w)
    square = np.zeros((size, size))
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = digit

    # 6. resize to 20×20
    digit_20 = cv2.resize(
        square, (20, 20), interpolation=cv2.INTER_AREA
    )

    # 7. place in center of 28×28
    img_28 = np.zeros((28, 28))
    img_28[4:24, 4:24] = digit_20

    # 8. MNIST normalize
    # img_28 = (img_28 - 0.1307) / 0.3081

    x = torch.tensor(img_28, dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(0)  # 1×1×28×28

    return x, img_28


def show_channels_horizontal(tensor, title):
    """
    tensor: (1, N, H, W)
    """
    st.subheader(title)
    n_channels = tensor.shape[1]
    cols = st.columns(n_channels)
    for i in range(n_channels):
        img = tensor[0, i].detach().cpu().numpy()
        cols[i].image(img, clamp=True, width=70)


def show_liner(tensor, title):
    """
    tensor: (1, N)
    """
    st.subheader(title)
    vec = tensor[0].detach().cpu().numpy()
    img = np.repeat(vec[np.newaxis, :], 50, axis=0)
    st.image(img, clamp=True, width=800)


# ----------------------------
# Inference
# ----------------------------
if st.button("Submit"):
    if canvas.image_data is None:
        st.warning("Please draw a digit.")
    else:
        x, img = preprocess(canvas.image_data)

        st.subheader("Preprocessed Image")
        st.image(img, clamp=True, width=140)

        # 各層の出力を取得

        with torch.no_grad():
            out, h1, h2, h3, h4, h5 = model(x, True)
            probs = torch.softmax(out[0], dim=0)

        show_channels_horizontal(h1, "Conv1 + ReLU")
        show_channels_horizontal(h2, "Pool1")
        show_channels_horizontal(h3, "Conv2 + ReLU")
        show_channels_horizontal(h4, "Pool2")
        show_liner(h5, "Flatten")

        pred = torch.argmax(probs).item()

        st.subheader(f"Prediction: {pred}")

        st.write("Probability distribution")
        st.bar_chart(probs.numpy())

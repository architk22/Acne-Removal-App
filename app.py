import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from diffusers import StableDiffusionInpaintPipeline

from transunet import TransUNet

# ============================
# CONFIG
# ============================
IMG_SIZE = 224
MODEL_PATH = "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = "healthy clear skin, smooth natural skin texture, photorealistic"
NEGATIVE_PROMPT = "acne, blemish, scar, unrealistic, fake"

INPAINT_SIZE = 512

# ============================
# LOAD MODELS
# ============================
@st.cache_resource
def load_models():
    seg_model = TransUNet(num_classes=2, img_size=IMG_SIZE)
    seg_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    seg_model.to(DEVICE)
    seg_model.eval()

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None
    ).to(DEVICE)

    return seg_model, pipe

seg_model, pipe = load_models()

# ============================
# PREPROCESS
# ============================
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# ============================
# FUNCTIONS
# ============================
def get_mask(img, threshold, kernel_size, iterations):
    h, w = img.shape[:2]

    t = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = seg_model(t)

    prob = torch.softmax(logits, dim=1)[0,1].cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.dilate(mask, kernel, iterations=iterations)

    return Image.fromarray(mask * 255)

def scale_prompt(prompt, strength):
    return (prompt + ", ") * int(max(1, strength * 2))

def inpaint(img, mask, steps, guidance, prompt):
    img_pil = Image.fromarray(img).resize((INPAINT_SIZE, INPAINT_SIZE))
    mask = mask.resize((INPAINT_SIZE, INPAINT_SIZE))

    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=img_pil,
        mask_image=mask,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]

    return np.array(result.resize((img.shape[1], img.shape[0])))

# ============================
# UI
# ============================
st.title("Acne Removal (Advanced Controls)")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

# 🔥 PARAMETER CONTROLS
st.sidebar.header("Controls")

threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5)
kernel = st.sidebar.slider("Mask Kernel Size", 3, 21, 9, step=2)
iterations = st.sidebar.slider("Mask Iterations", 1, 5, 2)

guidance = st.sidebar.slider("Guidance Scale", 1.0, 15.0, 7.5)
steps = st.sidebar.slider("Inference Steps", 5, 50, 30)

prompt_strength = st.sidebar.slider("Prompt Strength", 0.5, 2.0, 1.0)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    st.image(img, caption="Original")

    if st.button("Run"):
        with st.spinner("Processing..."):

            mask = get_mask(img_np, threshold, kernel, iterations)

            if np.sum(np.array(mask)) == 0:
                st.warning("No acne detected")
                result = img_np
            else:
                prompt_scaled = scale_prompt(PROMPT, prompt_strength)
                result = inpaint(img_np, mask, steps, guidance, prompt_scaled)

        st.image(mask, caption="Mask")
        st.image(result, caption="Result")

        import io

        # ============================
        # MASK DOWNLOAD
        # ============================
        mask_pil = mask.convert("RGB") if isinstance(mask, Image.Image) else Image.fromarray(mask)

        mask_buf = io.BytesIO()
        mask_pil.save(mask_buf, format="PNG")

        st.download_button(
            label="Download Mask",
            data=mask_buf.getvalue(),
            file_name="mask.png",
            mime="image/png"
        )

        # ============================
        # RESULT DOWNLOAD
        # ============================
        result_pil = Image.fromarray(result)

        result_buf = io.BytesIO()
        result_pil.save(result_buf, format="PNG")

        st.download_button(
            label="Download Result Image",
            data=result_buf.getvalue(),
            file_name="acne_removed.png",
            mime="image/png"
        )
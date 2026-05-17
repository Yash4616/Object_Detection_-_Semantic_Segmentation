import gradio as gr
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from unet_model import UNet

# ---------------------------------------------------------
# 1. SETUP & MODEL LOADING
# ---------------------------------------------------------

# Ensure CPU deployment for HF Free Tier 
device = torch.device('cpu')

# 1A. Load YOLOv8 for Object Detection
yolo_model = YOLO("weights/best.pt")

# 1B. Load Custom PyTorch U-Net for Semantic Segmentation
unet_model = UNet(in_channels=3, num_classes=11)
state_dict_raw = torch.load("weights/last.pt", map_location=device, weights_only=False)

# Clean out any multi-GPU wrappers or nestings if present
if 'state_dict' in state_dict_raw:
    state_dict_raw = state_dict_raw['state_dict']
    
clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict_raw.items()}
unet_model.load_state_dict(clean_state_dict)
unet_model.eval()

# ---------------------------------------------------------
# 2. CONSTANTS & PROCESSING UTILS
# ---------------------------------------------------------

# Color palette for 11 mask classes (Class 0: Background, Classes 1-10: Digits 0-9)
MASK_COLORS = np.array([
    [0, 0, 0],         # 0: Background (Black)
    [228, 26, 28],     # 1: Red
    [55, 126, 184],    # 2: Blue
    [77, 175, 74],     # 3: Green
    [152, 78, 163],    # 4: Purple
    [255, 127, 0],     # 5: Orange
    [255, 255, 51],    # 6: Yellow
    [166, 86, 40],     # 7: Brown
    [247, 129, 191],   # 8: Pink
    [153, 153, 153],   # 9: Gray
    [102, 194, 165],   # 10: Teal
], dtype=np.uint8)

# Transform pipelining for the U-Net specific normalizations
unet_transform = transforms.Compose([
    transforms.ToTensor(), # converts img to [0.0, 1.0] and shifts format to (C, H, W)
])

def process_canvas(canvas_data, yolo_conf, yolo_iou):
    """
    Simultaneously process the dual-track DL pipeline from sketch canvas to model outputs.
    """
    if canvas_data is None or "composite" not in canvas_data:
        return None, None
        
    # Grab the completed drawing layer from sketch pad data dict
    sketched_img = canvas_data["composite"]
    
    # Strip transparency/alpha channels, force standard RGB
    rgb_img = sketched_img.convert("RGB")
    
    # Resize down exactly to 128x128 bounding resolution for models
    processed_img = rgb_img.resize((128, 128), Image.Resampling.BILINEAR)
    
    # Base Numpy array for YOLO
    img_np = np.array(processed_img)
    
    # --- PIPELINE 1: YOLOv8 Object Detection ---
    results = yolo_model.predict(source=img_np, conf=yolo_conf, iou=yolo_iou, verbose=False)
    yolo_rendered = results[0].plot() # Gets the overlay array with bounding boxes
    
    # --- PIPELINE 2: Custom U-Net Semantic Segmentation ---
    img_tensor = unet_transform(processed_img).unsqueeze(0).to(device) # Shape: [1, 3, 128, 128]
    
    with torch.no_grad():
        logits = unet_model(img_tensor) # Shape: [1, 11, 128, 128]
        # Determine the maximum probability class along dimension 1 (number of channels)
        argmax_preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy() # Shape: [128, 128]
        
    # Map the resulting 2D argmax array to RGB using our color array
    unet_rendered = MASK_COLORS[argmax_preds]
    
    return yolo_rendered, unet_rendered

# ---------------------------------------------------------
# 3. GRADIO USER INTERFACE LAYOUT
# ---------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# MNISTDD-RGB Sandbox: Dual-Track Detection & Segmentation Pipeline")
    gr.Markdown("An interactive deployment environment hosting parallel YOLOv8 (Detection) and Custom U-Net (Segmentation) inferences on real-time drawings mapped down to 128x128.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input sketching canvas
            canvas = gr.Sketchpad(
                type="pil", 
                label="Draw Overlapping Digits Here",
                brush=gr.Brush(colors=["#ffffff"]) # typical white brush mapping
            )
            
            with gr.Accordion("YOLOv8 Parameters", open=True):
                conf_slider = gr.Slider(minimum=0.10, maximum=1.00, value=0.25, step=0.01, label="YOLO Box Confidence Threshold")
                iou_slider = gr.Slider(minimum=0.10, maximum=1.00, value=0.45, step=0.01, label="NMS Overlap Threshold")
                
            trigger_button = gr.Button("Evaluate Pipeline", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Row():
                yolo_output = gr.Image(label="YOLOv8 Localized Target Extraction", type="numpy", interactive=False)
                unet_output = gr.Image(label="U-Net Pixel Classification Mask", type="numpy", interactive=False)
                
    trigger_button.click(
        fn=process_canvas,
        inputs=[canvas, conf_slider, iou_slider],
        outputs=[yolo_output, unet_output]
    )

if __name__ == "__main__":
    app.launch()

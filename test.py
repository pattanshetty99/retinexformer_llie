import os
import torch
import cv2
from tqdm import tqdm
from models.retinexformer import RetinexFormer

# ==========================
# CONFIG
# ==========================
INPUT_DIR = "test_images"
OUTPUT_DIR = "results"
MODEL_PATH = "checkpoints/best_model.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# LOAD MODEL
# ==========================
model = RetinexFormer().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ==========================
# INFERENCE LOOP
# ==========================
image_list = sorted(os.listdir(INPUT_DIR))

for img_name in tqdm(image_list):

    img_path = os.path.join(INPUT_DIR, img_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    tensor = torch.FloatTensor(img).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    output = output.squeeze().permute(1,2,0).cpu().numpy()
    output = (output * 255).clip(0,255).astype("uint8")
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), output)

print("Testing complete. Results saved.")

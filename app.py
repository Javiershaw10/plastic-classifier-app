import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# --------------------------------------------------
# 1. Setup FastAPI
# --------------------------------------------------
app = FastAPI(title="Plastic Classification API",
              description="Classify plastic types: HDPE, PET, PP, PS",
              version="1.0")

# Enable CORS if this API will be used by web/mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 2. Load Model
# --------------------------------------------------
labels = ["HDPE", "PET", "PP", "PS"]

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, len(labels))

model.load_state_dict(torch.load("plastic_classifier_mobilenetv2.pth", map_location="cpu"))
model.eval()

# --------------------------------------------------
# 3. Image Transform
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------------------------------------
# 4. Predict Function
# --------------------------------------------------
def predict(image: Image.Image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    return labels[pred.item()]

# --------------------------------------------------
# 5. Routes / Endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "Plastic Classification API is running!"}

@app.post("/predict")
async def classify_plastic(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = predict(image)
    return {
        "filename": file.filename,
        "prediction": result
    }

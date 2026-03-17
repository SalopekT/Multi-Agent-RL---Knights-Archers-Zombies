import data_generator
from ultralytics import YOLO
model = YOLO("C:\\Users\\tinsa\\KULeuven\\ml-project-2025-2026-main\\ml-project-2025-2026-main\\weights_vision3\\best (5).pt")
# Example image
img_path = "C:\\Users\\tinsa\\KULeuven\\ml-project-2025-2026-main\\ml-project-2025-2026-main\\dataset2\\images\\train\\img4067.jpeg"

# Run inference
results = model.predict(source=img_path, imgsz=416)

# Show image with predictions
results[0].show()
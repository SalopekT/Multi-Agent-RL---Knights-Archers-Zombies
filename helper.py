from ultralytics import YOLO

model = YOLO("weights_vision/best(3).pt")
results = model.predict("dataset/images/val/img951.jpeg")
print(results)
results[0].save("predicted_image4.jpg")
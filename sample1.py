from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
from PIL import Image
from codecarbon import OfflineEmissionsTracker

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

print(image)

image1 = Image.open("image1.jpg")
print(image1)

tracker = OfflineEmissionsTracker(country_iso_code="FRA")
tracker.start_task("load model")

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
tracker.stop_task()

tracker.start_task("compute")
inputs = processor(image1, return_tensors="pt")
tracker.stop_task()
tracker.stop()

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])

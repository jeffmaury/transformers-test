# Use a pipeline as a high-level helper
from transformers import pipeline
from codecarbon import OfflineEmissionsTracker

tracker = OfflineEmissionsTracker(country_iso_code="FRA")
tracker.start_task("load model")
pipe = pipeline("text-generation", model="bigcode/starcoder")
tracker.stop_task()

tracker.start_task("compute")
result = pipe("def print_hello_world():")
tracker.stop_task();
tracker.stop()
print(result)
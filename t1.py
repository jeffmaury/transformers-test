# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("sentiment-analysis")
result = pipe("this move is really crap")
print(result)
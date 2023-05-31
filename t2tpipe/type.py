from typing import TypedDict

class TextSampleForInference(TypedDict):
    x: str

class TextSampleForTrain(TypedDict):
    x: str
    y: str

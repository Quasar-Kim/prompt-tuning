from typing import TypedDict


class TextSampleForPrediction(TypedDict):
    x: str


class TextSampleForTrain(TypedDict):
    x: str
    y: str

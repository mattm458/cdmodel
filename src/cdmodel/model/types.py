from typing import Literal

AttentionActivation = Literal["sigmoid"] | Literal["softmax"]
AttentionMaskingStrategy = Literal["partner"] | Literal["both"]
FeatureFormat = Literal["feature"] | Literal["feature_delta"]
EmbeddingInputs = list[
    Literal["encoder"] | Literal["decoder"] | Literal["attention"] | Literal["linear"]
]
SpeakerInputs = list[Literal["encoder"] | Literal["decoder"] | Literal["attention"]]
IstInputs = list[
    Literal["encoder"] | Literal["decoder"] | Literal["attention"] | Literal["linear"]
]

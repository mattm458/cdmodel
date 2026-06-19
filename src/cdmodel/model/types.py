from typing import Literal

AttentionActivation = Literal["sigmoid"] | Literal["softmax"]
AttentionMaskingStrategy = Literal["self"] | Literal["partner"] | Literal["both"]
FeatureFormat = Literal["feature"] | Literal["feature_delta"]
EmbeddingInputs = list[
    Literal["encoder"] | Literal["decoder"] | Literal["attention"] | Literal["linear"]
]
SpeakerInputs = list[Literal["encoder"] | Literal["decoder"] | Literal["attention"]]
SpeakerSexInputs = list[
    Literal["encoder"] | Literal["decoder"] | Literal["attention"] | Literal["linear"]
]
IstInputs = list[
    Literal["encoder"]
    | Literal["history"]
    | Literal["decoder"]
    | Literal["attention"]
    | Literal["linear"]
]

import math
from collections import OrderedDict
from enum import Enum
from typing import Final, Literal, NamedTuple, Optional, Union

import torch
from lightning import pytorch as pl
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.common.data import ConversationData, Role
from cdmodel.common.model import ISTSides, ISTStyle
from cdmodel.common.role_assignment import AgentPartnerPredictionRole, PredictionType
from cdmodel.model.components.components import (
    AttentionActivation,
    Decoder,
    DualAttention,
    EmbeddingEncoder,
    Encoder,
    NoopAttention,
    SingleAttention,
    SinglePartnerAttention,
)
from cdmodel.model.components.ext_ist import ISTIncrementalEncoder, ISTOneShotEncoder
from cdmodel.model.util.ext_ist import (
    ext_ist_incremental_encode,
    ext_ist_one_shot_encode,
    ext_ist_validate,
)
from cdmodel.model.util.util import one_hot_drop_0, timestep_split


def expand_cat(x: list[Tensor], dim: int = -1):
    num_dims = None
    max_len = -1
    for t in x:
        if num_dims is None:
            num_dims = len(t.shape)
        elif num_dims != len(t.shape):
            raise Exception("Tensors in x are not of uniform dimensionality!")
        if t.shape[dim] > max_len:
            max_len = t.shape[dim]

    if num_dims is None:
        raise Exception("Cannot expand_cat an empty list!")

    if dim == -1:
        dim = num_dims - 1

    padded: list[Tensor] = []
    for t in x:
        padding = [0, 0] * num_dims
        padding[-((dim * 2) + 1)] = max_len - t.shape[dim]
        p = F.pad(t, tuple(padding)).unsqueeze(dim)
        padded.append(p)

    return torch.cat(padded, dim=dim)


class CDModelOutput(NamedTuple):
    predicted_segment_features: Tensor
    # TODO: predict_next should come from the DataLoader now
    predict_next: Tensor
    a_scores: Tensor | None
    b_scores: Tensor | None
    combined_scores: Tensor | None
    a_mask: Tensor
    b_mask: Tensor
    ist_weights_one_shot: dict[Role, Tensor] | None
    ist_embeddings: Tensor | None


# TODO: Should these be moved to a common source file?
CDPredictionStrategy = Enum("CDPredictionStrategy", ["both", "agent"])
SpeakerRoleEncoding = Enum("SpeakerRoleEncoding", ["one_hot"])
CDAttentionStyle = Enum("CDAttentionStyle", ["dual", "single_both", "single_partner"])
ISTAttentionStyle = Enum("ISTAttentionStyle", ["multi_head", "additive"])
CDModelPredictFormat = Enum("CDModelPredictFormat", ["value", "delta"])


class CDModel(pl.LightningModule):
    def __init__(
        self,
        feature_names: list[str],
        embedding_dim: int,
        embedding_encoder_out_dim: int,
        embedding_encoder_num_layers: int,
        embedding_encoder_dropout: float,
        embedding_encoder_att_dim: int,
        use_embeddings: bool,
        encoder_hidden_dim: int,
        encoder_num_layers: int,
        encoder_dropout: float,
        decoder_att_dim: int,
        decoder_hidden_dim: int,
        decoder_num_layers: int,
        decoder_dropout: float,
        attention_style: str,
        encoder_speaker_role: bool,
        att_context_speaker_role: bool,
        att_weighting_strategy: Literal["att"] | Literal["random"] | Literal["uniform"],
        decoder_speaker_role: bool,
        num_decoders: int,
        speaker_role_encoding: str,
        role_type: str,
        lr: float,
        predict_format: str,
        autoregressive_training: bool,
        autoregressive_inference: bool,
        ext_ist_enabled: bool,  # Whether to enable ISTs (see below)
        ext_ist_encoded_concat: bool = False,  # Whether to concatenate the IST with encoded input
        ext_ist_att_in: bool = False,  # Whether to pass the IST into the attention layer
        ext_ist_decoder_in: bool = False,  # Whether to pass the IST into the decoder
        ext_ist_sides: Optional[
            str
        ] = None,  # Whether we should compute the IST for one side ("single") or both sides ("both").
        ext_ist_style: Optional[
            str
        ] = None,  # Whether the IST should be computed at once from all conversational data ("one_shot"), computed incrementally as the conversation progresses ("incremental"), or both ("blended"). Blended mode is only available in dialogue system mode.
        ext_ist_objective_speaker_id: bool = False,  # Whether the IST token should be evaluated on its ability to predict the speaker ID during training
        ext_ist_objective_speaker_id_num: Optional[int] = None,
        ext_ist_token_dim: Optional[int] = None,  # The dimensionality of each IST token
        ext_ist_token_count: Optional[int] = None,  # The number of IST tokens
        ext_ist_encoder_dim: Optional[int] = None,  # The IST encoder output dimensions
        ext_ist_att_activation: Optional[str] = None,
        ext_ist_use_feature_deltas: bool = True,
        ext_ist_use_feature_values: bool = False,
        ext_ist_tanh_pre: bool = True,  # Whether to apply tanh activation to IST tokens before attention
        ext_ist_tanh_post: bool = False,  # Whether to apply tanh activation to IST embeddings after attention
        ext_ist_offset: Optional[
            Tensor
        ] = None,  # Optional offsets to apply to IST weights for testing
        embedding_type: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if use_embeddings and embedding_type not in {"segment", "word"}:
            raise Exception("embedding_type must be one of either 'segment' or 'word")

        self.lr: Final[float] = lr

        self.feature_names: Final[list[str]] = feature_names
        self.num_features: Final[int] = len(feature_names)

        self.output_format: Final[CDModelPredictFormat] = CDModelPredictFormat[
            predict_format
        ]
        del predict_format

        self.autoregressive_training: Final[bool] = autoregressive_training
        self.autoregressive_inference: Final[bool] = autoregressive_inference

        self.speaker_role_encoding: Final[SpeakerRoleEncoding] = SpeakerRoleEncoding[
            speaker_role_encoding
        ]
        del speaker_role_encoding
        self.role_type: Final[PredictionType] = PredictionType[role_type]
        del role_type
        self.attention_style: Final[CDAttentionStyle] = CDAttentionStyle[
            attention_style
        ]
        del attention_style

        self.encoder_speaker_role: Final[bool] = encoder_speaker_role
        self.att_context_speaker_role: Final[bool] = att_context_speaker_role
        self.decoder_speaker_role: Final[bool] = decoder_speaker_role

        self.ext_ist_sides: Final[ISTSides | None] = (
            ISTSides[ext_ist_sides] if ext_ist_sides is not None else None
        )
        del ext_ist_sides

        self.ext_ist_style: Final[ISTStyle | None] = (
            ISTStyle[ext_ist_style] if ext_ist_style is not None else None
        )
        del ext_ist_style

        self.ext_ist_att_activation: Final[AttentionActivation | None] = (
            AttentionActivation[ext_ist_att_activation]
            if ext_ist_att_activation is not None
            else None
        )
        del ext_ist_att_activation

        self.ext_ist_offset: Final[Tensor | None] = ext_ist_offset

        # Embedding Encoder
        # =====================
        # The embedding encoder encodes textual data associated with each conversational
        # segment. At each segment, it accepts a sequence of word embeddings and outputs
        # a vector of size `embedding_encoder_out_dim`.
        self.use_embeddings: Final[bool] = use_embeddings
        self.embedding_type: Final[str | None] = embedding_type

        self.embedding_encoder: EmbeddingEncoder | None = None
        self.embedding_linear = None

        if self.use_embeddings and self.embedding_type == "word":
            self.embedding_encoder = EmbeddingEncoder(
                embedding_dim=embedding_dim,
                encoder_out_dim=embedding_encoder_out_dim,
                encoder_num_layers=embedding_encoder_num_layers,
                encoder_dropout=embedding_encoder_dropout,
                attention_dim=embedding_encoder_att_dim,
            )
        elif self.use_embeddings and self.embedding_type == "segment":
            # TODO: Make this configurable
            self.embedding_linear = nn.Sequential(nn.Linear(768, 50), nn.Tanh())

        # Segment encoder
        # =====================
        # The segment encoder outputs a representation of each conversational segment. The
        # encoder input includes speech features extracted from the segment, an encoded
        # representation of the words spoken in the segment, and a one-hot vector of
        # the speaker role. Each encoded segment is kept by appending it to the conversation
        # history.
        encoder_in_dim: int = (
            self.num_features  # The number of speech features the model is predicting
            + (2 if self.encoder_speaker_role else 0)  # One-hot speaker role vector
        )

        if self.use_embeddings:
            encoder_in_dim += (
                embedding_encoder_out_dim  # Dimensions of the embedding encoder output
            )

        # Main encoder for input features
        print(f"Encoder: encoder_hidden_dim = {encoder_hidden_dim}")
        self.encoder_hidden_dim: Final[int] = encoder_hidden_dim
        self.encoder = Encoder(
            in_dim=encoder_in_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
        )

        # The dimensions of each historical timestep as output by the segment encoder.
        history_dim: Final[int] = encoder_hidden_dim + (
            ext_ist_token_dim
            if ext_ist_enabled
            and ext_ist_encoded_concat
            and ext_ist_token_dim is not None
            else 0
        )

        # Attention
        # =====================
        # The attention mechanisms attend to segments in the conversation history. They
        # determine which segments are most useful for decoding into the upcoming speech
        # features. Depending on the configuration, there may be one or more attention
        # mechanisms and one or more decoders.

        # Each attention mechanism outputs a tensor of the same size as a historical
        # timestep. Calculate the total output size depending on whether
        # we're using dual or single attention
        att_multiplier: Final[int] = 2 if self.attention_style == "dual" else 1
        att_history_out_dim: Final[int] = history_dim * att_multiplier

        ext_ist_att_context_dim: int = (
            ext_ist_token_dim
            if ext_ist_enabled and ext_ist_att_in and ext_ist_token_dim is not None
            else 0
        )
        if self.ext_ist_sides == ISTSides.both:
            ext_ist_att_context_dim *= 2

        att_context_dim: int = (
            +(decoder_hidden_dim * decoder_num_layers)  # The decoder hidden state
            + (ext_ist_att_context_dim)  # IST token dimensions if active
            + (2 if self.att_context_speaker_role else 0)  # One-hot speaker role vector
        )

        if self.use_embeddings:
            att_context_dim += embedding_encoder_att_dim  # The encoded representation of the upcoming segment transcript

        # Initialize the attention mechanisms
        if self.attention_style == CDAttentionStyle.dual:
            self.attentions = nn.ModuleList(
                [
                    DualAttention(
                        history_in_dim=history_dim,
                        context_dim=att_context_dim,
                        att_dim=decoder_att_dim,
                        weighting_strategy=att_weighting_strategy,
                    )
                    for _ in range(num_decoders)
                ]
            )
        elif self.attention_style == CDAttentionStyle.single_both:
            self.attentions = nn.ModuleList(
                [
                    SingleAttention(
                        history_in_dim=history_dim,
                        context_dim=att_context_dim,
                        att_dim=decoder_att_dim,
                        weighting_strategy=att_weighting_strategy,
                    )
                    for _ in range(num_decoders)
                ]
            )
        elif self.attention_style == CDAttentionStyle.single_partner:
            self.attentions = nn.ModuleList(
                [
                    SinglePartnerAttention(
                        history_in_dim=history_dim,
                        context_dim=att_context_dim,
                        att_dim=decoder_att_dim,
                        weighting_strategy=att_weighting_strategy,
                    )
                    for _ in range(num_decoders)
                ]
            )
        # TODO: Make None attention explicitly a CDAttentionStyle enum value
        elif self.attention_style is None:
            self.attentions = nn.ModuleList(
                [NoopAttention() for _ in range(num_decoders)]
            )
        else:
            raise Exception(f"Unrecognized attention style '{self.attention_style}'")

        # Decoders
        # =====================
        # Decoders predict speech features from the upcoming segment based on its transcript
        # and from attention-summarized historical segments.

        ext_ist_decoder_in_dim: int = (
            ext_ist_token_dim
            if ext_ist_enabled and ext_ist_decoder_in and ext_ist_token_dim is not None
            else 0
        )
        if self.ext_ist_sides == ISTSides.both:
            ext_ist_decoder_in_dim *= 2

        decoder_in_dim = (
            att_history_out_dim  # Size of the attention mechanism output for summarized historical segments
            + (2 if self.decoder_speaker_role else 0)  # One-hot speaker role vector
            + ext_ist_decoder_in_dim  # IST token dimensions if active
        )

        if self.use_embeddings:
            decoder_in_dim += embedding_encoder_out_dim  # Size of the embedding encoder output for upcoming segment text

        # Initialize the decoders
        if num_decoders == 1:
            decoder_out_dim = len(feature_names)
        elif num_decoders == len(feature_names):
            decoder_out_dim = 1
        else:
            # TODO: Make this a more specific execption type
            raise Exception(
                f"Configuration specifies {num_decoders} which cannot output {len(feature_names)} output features!"
            )

        self.decoders = nn.ModuleList(
            [
                Decoder(
                    decoder_in_dim=decoder_in_dim,
                    hidden_dim=decoder_hidden_dim,
                    num_layers=decoder_num_layers,
                    decoder_dropout=decoder_dropout,
                    output_dim=decoder_out_dim,
                    activation=None,
                )
                for _ in range(num_decoders)
            ]
        )

        # Extensions
        # =====================

        # Interaction Style Tokens (IST)
        # ---
        # Based on Tacotron style tokens from Wang et al (2018).
        # Creates a bank of tokens trained to be associated with different conversational interaction
        # styles as expressed through turn-level feature deltas. A feature delta encoder predicts
        # weights for each of the tokens, which are summed into an interaction style embedding.
        # Currently, the interaction style embedding is given as context to the attention layer, but
        # not the decoder. The goal is to investigate whether different interaction styles can influence
        # which historical turns the model finds most useful.
        #
        # Currently, our IST experiments are related to determining the interaction style of an agent
        # in a dialogue system configuration. That is, the interaction style embedding is meant to
        # explain the "personality" of the agent. In the future, we may experimen with generating a
        # separate token for the partner, both to influence the partner's personality when the model is
        # in analytical mode, and to determine if the model can account for its partner's personality
        # when predicting its own speech features.
        self.ext_ist_enabled: Final[bool] = ext_ist_enabled
        self.ext_ist_encoded_concat: Final[bool] = ext_ist_encoded_concat
        self.ext_ist_att_in: Final[bool] = ext_ist_att_in
        self.ext_ist_decoder_in: Final[bool] = ext_ist_decoder_in

        self.ext_ist_objective_speaker_id: Final[bool] = ext_ist_objective_speaker_id

        self.ext_ist_encoder_one_shot: ISTOneShotEncoder | None = None
        self.ext_ist_encoder_incremental: ISTIncrementalEncoder | None = None
        self.ext_ist_speaker_id_linear: nn.Sequential | None = None
        self.ext_ist_gender_linear: nn.Sequential | None = None

        if ext_ist_enabled:
            ext_ist_validate(
                role_type=self.role_type,
                ist_sides=self.ext_ist_sides,
                ist_style=self.ext_ist_style,
            )

            # TODO: Clean up these requirements - I added a few new configuration options but they
            # aren't being covered here
            if ext_ist_token_count is None:
                raise Exception(
                    "If ISTs are enabled, ext_ist_token_count cannot be None"
                )
            if ext_ist_token_dim is None:
                raise Exception("If ISTs are enabled, ext_ist_token_dim cannot be None")
            if ext_ist_encoder_dim is None:
                raise Exception(
                    "If ISTs are enabled, ext_ist_encoder_dim cannot be None"
                )

            if (
                self.ext_ist_style == ISTStyle.one_shot
                or self.ext_ist_style == ISTStyle.blended
            ):
                self.ext_ist_encoder_one_shot = ISTOneShotEncoder(
                    token_count=ext_ist_token_count,
                    token_dim=ext_ist_token_dim,
                    num_features=self.num_features,
                    att_dim=ext_ist_token_dim,
                    att_activation=self.ext_ist_att_activation,
                    ext_ist_use_feature_deltas=ext_ist_use_feature_deltas,
                    ext_ist_use_feature_values=ext_ist_use_feature_values,
                    ext_ist_tanh_pre=ext_ist_tanh_pre,
                    ext_ist_tanh_post=ext_ist_tanh_post,
                )
            if self.ext_ist_style == ISTStyle.blended:
                self.ext_ist_encoder_incremental = ISTIncrementalEncoder(
                    token_count=ext_ist_token_count,
                    token_dim=ext_ist_token_dim,
                    num_features=self.num_features,
                    att_dim=ext_ist_token_dim,
                    att_activation=self.ext_ist_att_activation,
                    ext_ist_tanh_pre=ext_ist_tanh_pre,
                    ext_ist_tanh_post=ext_ist_tanh_post,
                )

            if ext_ist_objective_speaker_id and ext_ist_objective_speaker_id_num:
                self.ext_ist_speaker_id_linear = nn.Sequential(
                    nn.Linear(ext_ist_token_dim, ext_ist_objective_speaker_id_num),
                )

    # TODO: Keep the input parameters cleaned up
    def forward(
        self,
        segment_features: Tensor,
        segment_features_delta: Tensor,
        segment_features_sides: dict[Role, Tensor],
        segment_features_sides_len: dict[Role, list[int]],
        segment_features_delta_sides: dict[Role, Tensor],
        predict_next: Tensor,
        history_mask_a: Tensor,
        history_mask_b: Tensor,
        word_embeddings: Tensor | None,
        segment_embeddings: Tensor | None,
        embeddings_len: Tensor,
        conv_len: list[int],
        speaker_role_idx: Tensor,
        is_autoregressive: bool,
    ) -> CDModelOutput:
        # Get some basic information about the batch
        batch_size: Final[int] = segment_features.shape[0]
        num_segments: Final[int] = segment_features.shape[1]
        device = segment_features.device

        embeddings_encoded: Tensor | None = None
        if self.use_embeddings:
            if self.embedding_type == "word" and self.embedding_encoder is not None:
                embeddings_encoded, _ = self.embedding_encoder(
                    word_embeddings, embeddings_len
                )
                embeddings_encoded = nn.utils.rnn.pad_sequence(
                    torch.split(embeddings_encoded, conv_len), batch_first=True  # type: ignore
                )
            elif self.embedding_type == "segment":
                embeddings_encoded = self.embedding_linear(segment_embeddings)

        if self.speaker_role_encoding == SpeakerRoleEncoding.one_hot:
            speaker_role_encoded = one_hot_drop_0(speaker_role_idx, num_classes=3)
        else:
            raise NotImplementedError(
                f"Unsupported speaker role encoding {self.speaker_role_encoding}"
            )

        speaker_role_idx_segmented = timestep_split(speaker_role_idx)
        speaker_role_encoded_segmented = timestep_split(speaker_role_encoded)
        features_arr = timestep_split(segment_features)
        segment_features_delta_segmented = timestep_split(segment_features_delta)

        embeddings_encoded_segmented: list[Tensor] | None = None
        if embeddings_encoded is not None:
            embeddings_encoded_segmented = timestep_split(embeddings_encoded)

        features_input_arr: list[Tensor] = []

        # Create initial zero hidden states for the encoder and decoder(s)
        encoder_hidden: list[Tensor] = self.encoder.get_hidden(
            batch_size=batch_size, device=device
        )
        decoder_hidden: list[list[Tensor]] = [
            d.get_hidden(batch_size, device=device) for d in self.decoders
        ]

        # Lists to store accumulated conversation data from the main loop below
        history_tensor = torch.zeros(
            (batch_size, num_segments - 1, self.encoder_hidden_dim), device=device
        )
        decoded_all_cat: list[Tensor] = []

        a_scores_all: list[Tensor] = []
        b_scores_all: list[Tensor] = []
        combined_scores_all: list[Tensor] = []
        a_mask_all: list[Tensor] = []
        b_mask_all: list[Tensor] = []

        # Placeholders to contain predicted features carried over from the previous timestep
        features_prev = torch.zeros((batch_size, self.num_features), device=device)

        # IST Tokens
        ist_embeddings_one_shot: OrderedDict[Role, Tensor] | None = None
        ist_weights_one_shot: OrderedDict[Role, Tensor] | None = None
        if self.ext_ist_encoder_one_shot is not None:
            ist_embeddings_one_shot, ist_weights_one_shot = ext_ist_one_shot_encode(
                encoder=self.ext_ist_encoder_one_shot,
                role_type=self.role_type,
                ist_sides=self.ext_ist_sides,
                ist_style=self.ext_ist_style,
                segment_features_sides=segment_features_sides,
                segment_features_delta_sides=segment_features_delta_sides,
                segment_features_sides_len=segment_features_sides_len,
                offset=self.ext_ist_offset,
            )

        ext_ist_incremental_h: Tensor | None = None
        ext_ist_incremental_accumulator: Tensor | None = None
        if self.ext_ist_encoder_incremental is not None:
            ext_ist_incremental_h = self.ext_ist_encoder_incremental.get_hidden(
                batch_size=batch_size, device=device, precision="16-true"
            )
            ext_ist_incremental_accumulator = (
                self.ext_ist_encoder_incremental.get_accumulator(
                    batch_size=batch_size, device=device, precision="16-true"
                )
            )

        # Iterate through the conversation
        autoregress_mask_segmented = timestep_split(
            F.pad(predict_next[:, :-1], (1, 0)) & is_autoregressive
        )

        ist_embeddings: Tensor | None = None
        for i in range(num_segments - 1):
            ist_embeddings_cat: list[Tensor] = []
            if ist_embeddings_one_shot is not None:
                ist_embeddings_cat.extend(list(ist_embeddings_one_shot.values()))

            # Autoregression/teacher forcing:
            # Create a mask that contains True if a previously predicted feature should be fed
            # back into the model, or False if the ground truth value should be used instead
            # (i.e., teacher forcing)
            autoregress_mask_i = autoregress_mask_segmented[i]
            features_i = features_arr[i].clone()

            if self.output_format == CDModelPredictFormat.value:
                features_i[autoregress_mask_i] = (
                    features_prev[autoregress_mask_i]
                    .detach()
                    .clone()
                    .type(features_i.dtype)
                )
            elif self.output_format == CDModelPredictFormat.delta:
                if i > 0:
                    features_i[autoregress_mask_i] = features_input_arr[i - 1][
                        autoregress_mask_i
                    ] + features_prev[autoregress_mask_i].detach().type(
                        features_i.dtype
                    )
                else:
                    features_i[autoregress_mask_i] = (
                        features_prev[autoregress_mask_i]
                        .detach()
                        .clone()
                        .type(features_i.dtype)
                    )
            else:
                raise NotImplementedError(self.output_format)

            features_input_arr.append(features_i)

            # If we're using the incremental IST encoder, encode now
            if (
                self.ext_ist_encoder_incremental is not None
                and ext_ist_incremental_h is not None
                and ext_ist_incremental_accumulator is not None
            ):
                ext_ist_incremental_accumulator, ext_ist_incremental_h = (
                    ext_ist_incremental_encode(
                        encoder=self.ext_ist_encoder_incremental,
                        role_type=self.role_type,
                        ist_sides=self.ext_ist_sides,
                        ist_style=self.ext_ist_style,
                        speaker_role_idx=speaker_role_idx_segmented[i],
                        features_delta=segment_features_delta_segmented[i],
                        accumulator=ext_ist_incremental_accumulator,
                        h=ext_ist_incremental_h,
                    )
                )

                ist_embeddings_cat.append(ext_ist_incremental_accumulator)

            if len(ist_embeddings_cat) > 0:
                ist_embeddings = torch.cat(ist_embeddings_cat, dim=-1)

            # Assemble the encoder input. This includes the current conversation features
            # and the previously-encoded embeddings.
            encoder_in_arr = [features_i]

            if embeddings_encoded_segmented is not None:
                encoder_in_arr.append(embeddings_encoded_segmented[i])

            if self.encoder_speaker_role:
                encoder_in_arr.append(speaker_role_encoded_segmented[i])

            encoder_in: Tensor = torch.cat(
                encoder_in_arr,
                dim=-1,
            )

            # Encode the input and append it to the history.
            encoded, encoder_hidden = self.encoder(encoder_in, encoder_hidden)

            history_tensor[:, i, :] = encoded

            # Concatenate the history tensor and select specific batch indexes where we are predicting
            history = history_tensor[:, : i + 1, :]

            if self.ext_ist_encoded_concat and ist_embeddings is not None:
                history = torch.concat(
                    [
                        history,
                        ist_embeddings.unsqueeze(1).repeat((1, history.shape[1], 1)),
                    ],
                    dim=-1,
                )

            a_scores_cat: list[Tensor] = []
            b_scores_cat: list[Tensor] = []
            a_mask_cat: list[Tensor] = []
            b_mask_cat: list[Tensor] = []
            combined_scores_cat: list[Tensor] = []
            features_pred_cat: list[Tensor] = []

            self.attention_sides = "role"
            if self.attention_sides == "perspective":
                pass
                # predicting_speaker_id = speaker_id_idx[:, i + 1].unsqueeze(-1)
                # partner_speaker_id = speaker_id_partner_idx[:, i + 1].unsqueeze(-1)

                # speaker_id_idx_history = speaker_id_idx[:, : i + 1]
                # speaker_id_idx_history_is_nonzero = speaker_id_idx_history != 0

                # # Side A is all turns from the speaker being predicted
                # a_mask = (speaker_id_idx_history == predicting_speaker_id) & (
                #     speaker_id_idx_history_is_nonzero
                # )

                # # Side B is all turns not from the speaker being predicted
                # b_mask = (speaker_id_idx_history == partner_speaker_id) & (
                #     speaker_id_idx_history_is_nonzero
                # )
            elif self.attention_sides == "role":
                a_mask = history_mask_a[:, : i + 1]
                b_mask = history_mask_b[:, : i + 1]

            for decoder_idx, (attention, h, decoder) in enumerate(
                zip(
                    self.attentions,
                    decoder_hidden,
                    self.decoders,
                )
            ):
                attention_context: list[Tensor] = []
                attention_context.extend(h)
                if embeddings_encoded_segmented is not None:
                    attention_context.append(embeddings_encoded_segmented[i + 1])
                if self.att_context_speaker_role:
                    attention_context.append(speaker_role_encoded_segmented[i + 1])

                if self.ext_ist_att_in and ist_embeddings is not None:
                    attention_context.append(ist_embeddings)

                history_encoded, att_scores = attention(
                    history,
                    context=torch.cat(attention_context, dim=-1),
                    a_mask=a_mask,
                    b_mask=b_mask,
                )

                if att_scores.a_scores is not None:
                    a_scores_cat.append(att_scores.a_scores)
                if att_scores.b_scores is not None:
                    b_scores_cat.append(att_scores.b_scores)
                if att_scores.combined_scores is not None:
                    combined_scores_cat.append(att_scores.combined_scores)

                decoder_in_arr = [
                    history_encoded,
                ]

                if embeddings_encoded_segmented is not None:
                    decoder_in_arr.append(embeddings_encoded_segmented[i + 1])

                if self.decoder_speaker_role:
                    decoder_in_arr.append(
                        speaker_role_encoded_segmented[i + 1],
                    )
                if ist_embeddings is not None and self.ext_ist_decoder_in:
                    decoder_in_arr.append(ist_embeddings)
                decoder_in = torch.cat(decoder_in_arr, dim=-1)

                decoder_out, h_out, decoder_high = decoder(decoder_in, h)
                decoder_hidden[decoder_idx] = h_out
                features_pred_cat.append(decoder_out)

            a_mask_cat.append(a_mask)
            b_mask_cat.append(b_mask)

            # Assemble final predicted features
            features_pred = torch.cat(features_pred_cat, dim=-1)
            decoded_all_cat.append(features_pred.unsqueeze(1))

            if len(a_scores_cat) > 0:
                a_scores_all.append(torch.cat(a_scores_cat, dim=1))
            if len(b_scores_cat) > 0:
                b_scores_all.append(torch.cat(b_scores_cat, dim=1))
            if len(combined_scores_cat) > 0:
                combined_scores_all.append(torch.cat(combined_scores_cat, dim=1))

            a_mask_all.append(torch.cat(a_mask_cat, dim=1))
            b_mask_all.append(torch.cat(b_mask_cat, dim=1))

            features_prev = features_pred

        return CDModelOutput(
            predicted_segment_features=torch.cat(decoded_all_cat, dim=1),
            predict_next=predict_next,
            ist_embeddings=ist_embeddings,
            ist_weights_one_shot=(
                dict(ist_weights_one_shot) if ist_weights_one_shot is not None else None
            ),
            a_scores=(
                expand_cat(a_scores_all, dim=-1) if len(a_scores_all) > 0 else None
            ),
            b_scores=(
                expand_cat(b_scores_all, dim=-1) if len(b_scores_all) > 0 else None
            ),
            combined_scores=(
                expand_cat(combined_scores_all, dim=-1)
                if len(combined_scores_all) > 0
                else None
            ),
            a_mask=expand_cat(a_mask_all, dim=-1),
            b_mask=expand_cat(b_mask_all, dim=-1),
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch: ConversationData, batch_idx: int) -> Tensor:
        batch_size = batch.segment_features.shape[0]

        results = self(
            segment_features=batch.segment_features,
            segment_features_delta=batch.segment_features_delta,
            segment_features_sides=batch.segment_features_sides,
            segment_features_delta_sides=batch.segment_features_delta_sides,
            segment_features_sides_len=batch.segment_features_sides_len,
            predict_next=batch.predict_next,
            history_mask_a=batch.history_mask_a,
            history_mask_b=batch.history_mask_b,
            word_embeddings=batch.word_embeddings,
            segment_embeddings=batch.segment_embeddings,
            embeddings_len=batch.embeddings_len,
            conv_len=batch.num_segments,
            speaker_role_idx=batch.speaker_role_idx,
            is_autoregressive=self.autoregressive_training,
        )

        if batch_idx == 0 and results.ist_weights_one_shot is not None:
            for role, weights in results.ist_weights_one_shot.items():
                # Dimension 0: Batch size
                # Dimension 1: Number of tokens
                for weight_i in range(weights.shape[1]):
                    self.logger.experiment.add_histogram(
                        f"training_ext_ist_weight_{role}_{weight_i}",
                        weights[:, weight_i].detach(),
                        global_step=self.current_epoch,
                    )

        if self.output_format == CDModelPredictFormat.value:
            y = batch.segment_features[:, 1:]
        elif self.output_format == CDModelPredictFormat.delta:
            y = batch.segment_features_delta[:, 1:]
        else:
            raise NotImplementedError(
                f"Output format {self.output_format} not supported!"
            )

        # TODO: Formalize this - Are we checking both sides or just our own outputs?
        # max_len = max(batch.num_segments) - 1
        # mask = torch.arange(max_len).expand(batch_size, max_len) < torch.tensor(
        #     max_len
        # )  # .unsqueeze(2)
        mask = results.predict_next

        loss = F.mse_loss(
            results.predicted_segment_features[mask],
            y[mask],
        )

        self.log(
            "training_loss",
            loss.detach(),
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        if self.ext_ist_objective_speaker_id:
            speaker_id_loss = F.cross_entropy(
                results.ext_ist_pred_speaker_id,
                torch.tensor(
                    [
                        x[AgentPartnerPredictionRole.agent]
                        for x in batch.role_speaker_assignment_idx
                    ],
                    device=results.ext_ist_pred_speaker_id.device,
                ),
            )
            self.log(
                "training_speaker_id_loss",
                speaker_id_loss.detach(),
                on_epoch=True,
                on_step=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            loss += speaker_id_loss

        # if self.ext_ist_enabled:
        #     self.log(
        #         "training_ist_entropy",
        #         (
        #             torch.distributions.Categorical(
        #                 results.ist_weights.detach()
        #             ).entropy()
        #             / math.log(y.shape[0])
        #         ).mean(),
        #         on_epoch=True,
        #         on_step=True,
        #         sync_dist=True,
        #     )

        # for feature_idx, feature_name in enumerate(self.feature_names):
        #     self.log(
        #         f"training_loss_l1_{feature_name}",
        #         F.smooth_l1_loss(
        #             results.predicted_segment_features[results.predict_next][
        #                 :, feature_idx
        #             ],
        #             y[results.predict_next][:, feature_idx],
        #         ).detach(),
        #         on_epoch=True,
        #         on_step=True,
        #         sync_dist=True,
        #         batch_size=batch_size,
        #     )

        return loss

    def validation_step(self, batch: ConversationData, batch_idx: int) -> Tensor:
        batch_size = batch.segment_features.shape[0]

        results = self(
            segment_features=batch.segment_features,
            segment_features_delta=batch.segment_features_delta,
            segment_features_sides=batch.segment_features_sides,
            segment_features_delta_sides=batch.segment_features_delta_sides,
            segment_features_sides_len=batch.segment_features_sides_len,
            predict_next=batch.predict_next,
            history_mask_a=batch.history_mask_a,
            history_mask_b=batch.history_mask_b,
            word_embeddings=batch.word_embeddings,
            segment_embeddings=batch.segment_embeddings,
            embeddings_len=batch.embeddings_len,
            conv_len=batch.num_segments,
            speaker_role_idx=batch.speaker_role_idx,
            is_autoregressive=self.autoregressive_inference,
        )

        if batch_idx == 0 and results.ist_weights_one_shot is not None:
            for role, weights in results.ist_weights_one_shot.items():
                # Dimension 0: Batch size
                # Dimension 1: Number of tokens
                for weight_i in range(weights.shape[1]):
                    self.logger.experiment.add_histogram(
                        f"validation_ext_ist_weight_{role}_{weight_i}",
                        weights[:, weight_i],
                        global_step=self.current_epoch,
                    )

        if self.output_format == CDModelPredictFormat.value:
            y = batch.segment_features[:, 1:]
        elif self.output_format == CDModelPredictFormat.delta:
            y = batch.segment_features_delta[:, 1:]
        else:
            raise NotImplementedError(
                f"Output format {self.output_format} not supported!"
            )

        loss = F.mse_loss(
            results.predicted_segment_features[results.predict_next],
            y[results.predict_next],
        )
        loss_l1 = F.smooth_l1_loss(
            results.predicted_segment_features[results.predict_next],
            y[results.predict_next],
        )

        self.log(
            "validation_loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "validation_loss_l1",
            loss_l1,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        if self.ext_ist_objective_speaker_id:
            speaker_id_loss = F.cross_entropy(
                results.ext_ist_pred_speaker_id,
                torch.tensor(
                    [
                        x[AgentPartnerPredictionRole.agent]
                        for x in batch.role_speaker_assignment_idx
                    ],
                    device=results.ext_ist_pred_speaker_id.device,
                ),
            )
            self.log(
                "validation_speaker_id_loss",
                speaker_id_loss,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                batch_size=batch_size,
            )

        # for feature_idx, feature_name in enumerate(self.feature_names):
        #     self.log(
        #         f"validation_loss_l1_{feature_name}",
        #         F.smooth_l1_loss(
        #             results.predicted_segment_features[results.predict_next][
        #                 :, feature_idx
        #             ],
        #             y[results.predict_next][:, feature_idx],
        #         ),
        #         on_epoch=True,
        #         on_step=False,
        #         sync_dist=True,
        #         batch_size=batch_size,
        #     )

        # if batch_idx == 0:
        #     print(results.ist_weights[:10])

        return loss

    def predict_step(
        self, batch: ConversationData, batch_idx: int, dataloader_idx: int = 0
    ):
        results = self(
            segment_features=batch.segment_features,
            segment_features_delta=batch.segment_features_delta,
            segment_features_sides=batch.segment_features_sides,
            segment_features_delta_sides=batch.segment_features_delta_sides,
            segment_features_sides_len=batch.segment_features_sides_len,
            predict_next=batch.predict_next,
            history_mask_a=batch.history_mask_a,
            history_mask_b=batch.history_mask_b,
            word_embeddings=batch.word_embeddings,
            segment_embeddings=batch.segment_embeddings,
            embeddings_len=batch.embeddings_len,
            conv_len=batch.num_segments,
            speaker_role_idx=batch.speaker_role_idx,
            is_autoregressive=self.autoregressive_inference,
        )

        if self.output_format == CDModelPredictFormat.value:
            y = batch.segment_features[:, 1:]
        elif self.output_format == CDModelPredictFormat.delta:
            y = batch.segment_features_delta[:, 1:]
        else:
            raise NotImplementedError(
                f"Output format {self.output_format} not supported!"
            )

        loss = F.mse_loss(
            results.predicted_segment_features,  # [results.predict_next],
            y,  # [results.predict_next],
        )
        loss_l1 = F.smooth_l1_loss(
            results.predicted_segment_features,  # [results.predict_next],
            y,  # [results.predict_next],
        )

        return batch, loss, loss_l1, results

    def on_train_epoch_end(self):
        for name, parameter in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, parameter, global_step=self.global_step
            )

    # TODO: Does the model need a test_step?

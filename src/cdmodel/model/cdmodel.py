import math
from collections import OrderedDict
from enum import Enum
from typing import Final, NamedTuple, Optional

import torch
from lightning import pytorch as pl
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.common.data import ConversationData, Role
from cdmodel.common.model import ISTSides, ISTStyle
from cdmodel.common.role_assignment import DialogueSystemRole, RoleType
from cdmodel.model.components.components import (
    Attention,
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
    a_scores: Tensor
    b_scores: Tensor
    a_mask: Tensor
    b_mask: Tensor
    ist_weights_one_shot: dict[Role, Tensor]
    ist_embeddings: Tensor | None


# TODO: Should these be moved to a common source file?
CDPredictionStrategy = Enum("CDPredictionStrategy", ["both", "agent"])
SpeakerRoleEncoding = Enum("SpeakerRoleEncoding", ["one_hot"])
CDAttentionStyle = Enum("CDAttentionStyle", ["dual", "single_both", "single_partner"])
ISTAttentionStyle = Enum("ISTAttentionStyle", ["multi_head", "additive"])

_GENDER_DICT: Final[dict[str, int]] = {"f": 0, "m": 1}


class CDModel(pl.LightningModule):
    # TODO: Should enum arguments should be strings, which are then cast to the enum?
    def __init__(
        self,
        feature_names: list[str],
        prediction_strategy: str,
        embedding_dim: int,
        embedding_encoder_out_dim: int,
        embedding_encoder_num_layers: int,
        embedding_encoder_dropout: float,
        embedding_encoder_att_dim: int,
        encoder_hidden_dim: int,
        encoder_num_layers: int,
        encoder_dropout: float,
        decoder_att_dim: int,
        decoder_hidden_dim: int,
        decoder_num_layers: int,
        decoder_dropout: float,
        attention_style: str,
        num_decoders: int,
        speaker_role_encoding: str,
        role_type: str,
        lr: float,
        ext_ist_enabled: bool,  # Whether to enable ISTs (see below)
        ext_ist_encoded_concat: bool,  # Whether to concatenate the IST with encoded input
        ext_ist_att_in: bool,  # Whether to pass the IST into the attention layer
        ext_ist_decoder_in: bool,  # Whether to pass the IST into the decoder
        ext_ist_sides: str,  # Whether we should compute the IST for one side ("single") or both sides ("both").
        ext_ist_style: str,  # Whether the IST should be computed at once from all conversational data ("one_shot"), computed incrementally as the conversation progresses ("incremental"), or both ("blended"). Blended mode is only available in dialogue system mode.
        ext_ist_objective_speaker_id: bool = False,  # Whether the IST token should be evaluated on its ability to predict the speaker ID during training
        ext_ist_objective_speaker_gender: bool = False,  # Whether the IST token should be evaluated on its ability to predict the speaker gender during training
        ext_ist_objective_speaker_id_num: Optional[int] = None,
        ext_ist_token_dim: Optional[int] = None,  # The dimensionality of each IST token
        ext_ist_token_count: Optional[int] = None,  # The number of IST tokens
        ext_ist_encoder_dim: Optional[int] = None,  # The IST encoder output dimensions
        ext_ist_att_activation: str = "softmax",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr: Final[float] = lr

        self.feature_names: Final[list[str]] = feature_names
        self.num_features: Final[int] = len(feature_names)

        self.speaker_role_encoding: Final[SpeakerRoleEncoding] = SpeakerRoleEncoding[
            speaker_role_encoding
        ]
        del speaker_role_encoding
        self.prediction_strategy: Final[CDPredictionStrategy] = CDPredictionStrategy[
            prediction_strategy
        ]
        del prediction_strategy
        self.role_type: Final[RoleType] = RoleType[role_type]
        del role_type
        self.attention_style: Final[CDAttentionStyle] = CDAttentionStyle[
            attention_style
        ]
        del attention_style

        self.ext_ist_sides: Final[ISTSides] = ISTSides[ext_ist_sides]
        del ext_ist_sides

        self.ext_ist_style: Final[ISTStyle] = ISTStyle[ext_ist_style]
        del ext_ist_style

        self.ext_ist_att_activation: Final[AttentionActivation] = AttentionActivation[
            ext_ist_att_activation
        ]
        del ext_ist_att_activation

        # Embedding Encoder
        # =====================
        # The embedding encoder encodes textual data associated with each conversational
        # segment. At each segment, it accepts a sequence of word embeddings and outputs
        # a vector of size `embedding_encoder_out_dim`.
        self.embedding_encoder = EmbeddingEncoder(
            embedding_dim=embedding_dim,
            encoder_out_dim=embedding_encoder_out_dim,
            encoder_num_layers=embedding_encoder_num_layers,
            encoder_dropout=embedding_encoder_dropout,
            attention_dim=embedding_encoder_att_dim,
        )

        # Segment encoder
        # =====================
        # The segment encoder outputs a representation of each conversational segment. The
        # encoder input includes speech features extracted from the segment, an encoded
        # representation of the words spoken in the segment, and a one-hot vector of
        # the speaker role. Each encoded segment is kept by appending it to the conversation
        # history.
        encoder_in_dim: Final[int] = (
            self.num_features  # The number of speech features the model is predicting
            + embedding_encoder_out_dim  # Dimensions of the embedding encoder output
            + 2  # One-hot speaker role vector
        )

        # Main encoder for input features
        print(f"Encoder: encoder_hidden_dim = {encoder_hidden_dim}")
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

        att_context_dim: Final[int] = (
            embedding_encoder_att_dim  # The encoded representation of the upcoming segment transcript
            + (decoder_hidden_dim * decoder_num_layers)  # The decoder hidden state
            + (ext_ist_att_context_dim)  # IST token dimensions if active
        )

        # Initialize the attention mechanisms
        if self.attention_style == CDAttentionStyle.dual:
            self.attentions = nn.ModuleList(
                [
                    DualAttention(
                        history_in_dim=history_dim,
                        context_dim=att_context_dim,
                        att_dim=decoder_att_dim,
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
            embedding_encoder_out_dim  # Size of the embedding encoder output for upcoming segment text
            + att_history_out_dim  # Size of the attention mechanism output for summarized historical segments
            + 2  # One-hot speaker role vector
            + ext_ist_decoder_in_dim  # IST token dimensions if active
        )

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
        self.ext_ist_objective_speaker_gender: Final[bool] = (
            ext_ist_objective_speaker_gender
        )

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

            self.ext_ist_tokens = nn.Parameter(
                torch.randn(ext_ist_token_count, ext_ist_token_dim), requires_grad=True
            )

            if (
                self.ext_ist_style == ISTStyle.one_shot
                or self.ext_ist_style == ISTStyle.blended
            ):
                self.ext_ist_encoder_one_shot = ISTOneShotEncoder(
                    token_dim=ext_ist_token_dim,
                    num_features=self.num_features,
                    att_dim=ext_ist_token_dim,
                    att_activation=self.ext_ist_att_activation,
                )
            if self.ext_ist_style == ISTStyle.blended:
                self.ext_ist_encoder_incremental = ISTIncrementalEncoder(
                    token_dim=ext_ist_token_dim,
                    num_features=self.num_features,
                    att_dim=ext_ist_token_dim,
                    att_activation=self.ext_ist_att_activation,
                )

            if ext_ist_objective_speaker_id and ext_ist_objective_speaker_id_num:
                self.ext_ist_speaker_id_linear = nn.Sequential(
                    nn.Linear(ext_ist_token_dim, ext_ist_objective_speaker_id_num),
                )

            if ext_ist_objective_speaker_gender:
                self.ext_ist_gender_linear = nn.Sequential(
                    nn.Linear(ext_ist_token_dim, 2)
                )

    # TODO: Keep the input parameters cleaned up
    def forward(
        self,
        segment_features: Tensor,
        segment_features_delta: Tensor,
        segment_features_delta_sides: dict[Role, Tensor],
        segment_features_delta_sides_len: dict[Role, list[int]],
        embeddings: Tensor,
        embeddings_len: Tensor,
        conv_len: list[int],
        role_speaker_assignment_idx: list[dict[Role, int]],
        speaker_role_idx: Tensor,
        is_autoregressive: bool,
    ) -> CDModelOutput:
        # Get some basic information about the batch
        batch_size: Final[int] = segment_features.shape[0]
        num_segments: Final[int] = segment_features.shape[1]
        device = segment_features.device

        embeddings_encoded, _ = self.embedding_encoder(embeddings, embeddings_len)
        embeddings_encoded = nn.utils.rnn.pad_sequence(
            torch.split(embeddings_encoded, conv_len), batch_first=True  # type: ignore
        )

        if self.speaker_role_encoding == SpeakerRoleEncoding.one_hot:
            speaker_role_encoded = one_hot_drop_0(speaker_role_idx, num_classes=3)
        else:
            raise NotImplementedError(
                f"Unsupported speaker role encoding {self.speaker_role_encoding}"
            )

        if self.role_type == RoleType.DialogueSystem:
            predict_next = (speaker_role_idx == DialogueSystemRole.agent.value)[:, 1:]
            a_history_mask = speaker_role_idx == DialogueSystemRole.agent.value
            b_history_mask = speaker_role_idx == DialogueSystemRole.partner.value
        else:
            raise NotImplementedError(f"Unsupported role type {self.role_type}")

        predict_next_segmented = timestep_split(predict_next)
        speaker_role_idx_segmented = timestep_split(speaker_role_idx)
        speaker_role_encoded_segmented = timestep_split(speaker_role_encoded)
        features_arr = timestep_split(segment_features)
        predict_next_segmented = timestep_split(predict_next)
        embeddings_encoded_segmented = timestep_split(embeddings_encoded)
        segment_features_delta_segmented = timestep_split(segment_features_delta)

        # Create initial zero hidden states for the encoder and decoder(s)
        encoder_hidden: list[Tensor] = self.encoder.get_hidden(
            batch_size=batch_size, device=device
        )
        decoder_hidden: list[list[Tensor]] = [
            d.get_hidden(batch_size, device=device) for d in self.decoders
        ]

        # Lists to store accumulated conversation data from the main loop below
        history_cat: list[Tensor] = []
        decoded_all_cat: list[Tensor] = []
        decoder_high_all_cat: list[Tensor] = []

        a_scores_all: list[Tensor] = []
        b_scores_all: list[Tensor] = []
        combined_scores_all: list[Tensor] = []
        a_mask_all: list[Tensor] = []
        b_mask_all: list[Tensor] = []

        # Placeholders to contain predicted features carried over from the previous timestep
        features_prev = torch.zeros((batch_size, self.num_features), device=device)
        predict_prev = torch.zeros((batch_size,), device=device, dtype=torch.bool)

        predict_cat: list[Tensor] = []

        history_encoded_timesteps: list[Tensor] = []

        # IST Tokens
        ist_embeddings_one_shot: OrderedDict[Role, Tensor] | None = None
        ist_weights_one_shot: OrderedDict[Role, Tensor] | None = None
        if self.ext_ist_encoder_one_shot is not None:
            ist_embeddings_one_shot, ist_weights_one_shot = ext_ist_one_shot_encode(
                encoder=self.ext_ist_encoder_one_shot,
                role_type=self.role_type,
                ist_sides=self.ext_ist_sides,
                ist_style=self.ext_ist_style,
                segment_features_delta_sides=segment_features_delta_sides,
                segment_features_delta_sides_len=segment_features_delta_sides_len,
                tokens=self.ext_ist_tokens,
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
        ist_embeddings: Tensor | None = None
        for i in range(num_segments - 1):
            predict_cat.append(predict_prev.unsqueeze(1))

            ist_embeddings_cat: list[Tensor] = []
            if ist_embeddings_one_shot is not None:
                ist_embeddings_cat.extend(list(ist_embeddings_one_shot.values()))

            # Autoregression/teacher forcing:
            # Create a mask that contains True if a previously predicted feature should be fed
            # back into the model, or False if the ground truth value should be used instead
            # (i.e., teacher forcing)
            autoregress_mask_i = predict_prev * is_autoregressive
            features_i = features_arr[i].clone()
            features_i[autoregress_mask_i] = (
                features_prev[autoregress_mask_i]
                .detach()
                .clone()
                .type(features_i.dtype)
            )

            # If we're using the incremental IST encoder, encode now
            if (
                self.ext_ist_encoder_incremental
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
                        tokens=self.ext_ist_tokens,
                    )
                )

                ist_embeddings_cat.append(ext_ist_incremental_accumulator)

            if len(ist_embeddings_cat) > 0:
                ist_embeddings = torch.cat(ist_embeddings_cat, dim=-1)

            # Assemble the encoder input. This includes the current conversation features
            # and the previously-encoded embeddings.
            encoder_in: Tensor = torch.cat(
                [
                    features_i,
                    embeddings_encoded_segmented[i],
                    speaker_role_encoded_segmented[i],
                ],
                dim=-1,
            )

            # Encode the input and append it to the history.
            encoded, encoder_hidden = self.encoder(encoder_in, encoder_hidden)

            history_cat.append(encoded)

            # Concatenate the history tensor and select specific batch indexes where we are predicting
            history = torch.stack(history_cat, dim=1)

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
            decoder_high_cat: list[Tensor] = []

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
                a_mask = a_history_mask[:, : i + 1]
                b_mask = b_history_mask[:, : i + 1]

            for decoder_idx, (attention, h, decoder) in enumerate(
                zip(
                    self.attentions,
                    decoder_hidden,
                    self.decoders,
                )
            ):
                attention_context = h + [embeddings_encoded_segmented[i + 1]]
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

                history_encoded_timesteps.append(history_encoded.detach().clone())

                decoder_in_arr = [
                    history_encoded,
                    embeddings_encoded_segmented[i + 1],
                    speaker_role_encoded_segmented[i + 1],
                ]
                if ist_embeddings is not None and self.ext_ist_decoder_in:
                    decoder_in_arr.append(ist_embeddings)
                decoder_in = torch.cat(decoder_in_arr, dim=-1)

                decoder_out, h_out, decoder_high = decoder(decoder_in, h)
                decoder_hidden[decoder_idx] = h_out
                features_pred_cat.append(decoder_out)
                decoder_high_cat.append(decoder_high.detach().clone())

            a_mask_cat.append(a_mask)
            b_mask_cat.append(b_mask)

            # Assemble final predicted features
            features_pred = torch.cat(features_pred_cat, dim=-1)
            decoder_high_all_cat.append(torch.cat(decoder_high_cat, dim=-1))
            decoded_all_cat.append(features_pred.unsqueeze(1))

            if len(a_scores_cat) > 0:
                a_scores_all.append(torch.cat(a_scores_cat, dim=1))
            if len(b_scores_cat) > 0:
                b_scores_all.append(torch.cat(b_scores_cat, dim=1))
            else:
                # TODO: Clarify what happens heres
                raise Exception("UH OH")
            if len(combined_scores_cat) > 0:
                combined_scores_all.append(torch.cat(combined_scores_cat, dim=1))

            a_mask_all.append(torch.cat(a_mask_cat, dim=1))
            b_mask_all.append(torch.cat(b_mask_cat, dim=1))

            predict_prev = predict_next_segmented[i]
            features_prev = features_pred

        predict_cat.append(predict_prev.unsqueeze(1))

        return CDModelOutput(
            predicted_segment_features=torch.cat(decoded_all_cat, dim=1),
            predict_next=predict_next,
            ist_embeddings=ist_embeddings,
            ist_weights_one_shot=dict(ist_weights_one_shot),
            a_scores=(
                expand_cat(a_scores_all, dim=-1) if len(a_scores_all) > 0 else None
            ),
            b_scores=(
                expand_cat(b_scores_all, dim=-1) if len(b_scores_all) > 0 else None
            ),
            a_mask=expand_cat(a_mask_all, dim=-1),
            b_mask=expand_cat(b_mask_all, dim=-1),
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch: ConversationData, batch_idx: int) -> Tensor:
        results = self(
            segment_features=batch.segment_features,
            segment_features_delta=batch.segment_features_delta,
            segment_features_delta_sides=batch.segment_features_delta_sides,
            segment_features_delta_sides_len=batch.segment_features_delta_sides_len,
            embeddings=batch.embeddings,
            embeddings_len=batch.embeddings_segment_len,
            conv_len=batch.num_segments,
            speaker_role_idx=batch.speaker_role_idx,
            role_speaker_assignment_idx=batch.role_speaker_assignment_idx,
            is_autoregressive=True,
        )

        y = batch.segment_features[:, 1:]
        loss = F.mse_loss(
            results.predicted_segment_features[results.predict_next],
            y[results.predict_next],
        )

        self.log(
            "training_loss", loss.detach(), on_epoch=True, on_step=True, sync_dist=True
        )

        if self.ext_ist_objective_speaker_id:
            speaker_id_loss = F.cross_entropy(
                results.ext_ist_pred_speaker_id,
                torch.tensor(
                    [
                        x[DialogueSystemRole.agent]
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
            )
            loss += speaker_id_loss

        if self.ext_ist_objective_speaker_gender:
            speaker_gender_loss = F.cross_entropy(
                results.ext_ist_pred_gender,
                torch.tensor(
                    [
                        _GENDER_DICT[x[DialogueSystemRole.agent]]
                        for x in batch.role_gender_assignment
                    ],
                    device=y.device,
                ),
            )
            self.log(
                "train_speaker_gender_loss",
                speaker_gender_loss.detach(),
                on_epoch=True,
                on_step=True,
                sync_dist=True,
            )
            loss += speaker_gender_loss

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

        for feature_idx, feature_name in enumerate(self.feature_names):
            self.log(
                f"training_loss_l1_{feature_name}",
                F.smooth_l1_loss(
                    results.predicted_segment_features[results.predict_next][
                        :, feature_idx
                    ],
                    y[results.predict_next][:, feature_idx],
                ).detach(),
                on_epoch=True,
                on_step=True,
                sync_dist=True,
            )

        return loss

    def validation_step(self, batch: ConversationData, batch_idx: int) -> Tensor:
        results = self(
            segment_features=batch.segment_features,
            segment_features_delta=batch.segment_features_delta,
            segment_features_delta_sides=batch.segment_features_delta_sides,
            segment_features_delta_sides_len=batch.segment_features_delta_sides_len,
            embeddings=batch.embeddings,
            embeddings_len=batch.embeddings_segment_len,
            conv_len=batch.num_segments,
            speaker_role_idx=batch.speaker_role_idx,
            role_speaker_assignment_idx=batch.role_speaker_assignment_idx,
            is_autoregressive=True,
        )

        y = batch.segment_features[:, 1:]
        loss = F.mse_loss(
            results.predicted_segment_features[results.predict_next],
            y[results.predict_next],
        )
        loss_l1 = F.smooth_l1_loss(
            results.predicted_segment_features[results.predict_next],
            y[results.predict_next],
        )

        self.log("validation_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(
            "validation_loss_l1", loss_l1, on_epoch=True, on_step=False, sync_dist=True
        )

        # if self.ext_ist_enabled:
        #     self.log(
        #         "validation_ist_entropy",
        #         (
        #             torch.distributions.Categorical(results.ist_weights).entropy()
        #             / math.log(y.shape[0])
        #         ).mean(),
        #         on_epoch=True,
        #         on_step=False,
        #         sync_dist=True,
        #     )

        if self.ext_ist_objective_speaker_id:
            speaker_id_loss = F.cross_entropy(
                results.ext_ist_pred_speaker_id,
                torch.tensor(
                    [
                        x[DialogueSystemRole.agent]
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
            )

        if self.ext_ist_objective_speaker_gender:
            speaker_gender_loss = F.cross_entropy(
                results.ext_ist_pred_gender,
                torch.tensor(
                    [
                        _GENDER_DICT[x[DialogueSystemRole.agent]]
                        for x in batch.role_gender_assignment
                    ],
                    device=y.device,
                ),
            )
            self.log(
                "validation_speaker_gender_loss",
                speaker_gender_loss,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

        for feature_idx, feature_name in enumerate(self.feature_names):
            self.log(
                f"validation_loss_l1_{feature_name}",
                F.smooth_l1_loss(
                    results.predicted_segment_features[results.predict_next][
                        :, feature_idx
                    ],
                    y[results.predict_next][:, feature_idx],
                ),
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

        # if batch_idx == 0:
        #     print(results.ist_weights[:10])

        return loss

    def predict_step(self, batch: ConversationData, batch_idx: int):
        return self(
            segment_features=batch.segment_features,
            segment_features_delta=batch.segment_features_delta,
            segment_features_delta_sides=batch.segment_features_delta_sides,
            segment_features_delta_sides_len=batch.segment_features_delta_sides_len,
            embeddings=batch.embeddings,
            embeddings_len=batch.embeddings_segment_len,
            conv_len=batch.num_segments,
            speaker_role_idx=batch.speaker_role_idx,
            role_speaker_assignment_idx=batch.role_speaker_assignment_idx,
            is_autoregressive=True,
        )

    def on_train_epoch_end(self):
        for name, parameter in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, parameter, global_step=self.global_step
            )

    # TODO: Does the model need a test_step?

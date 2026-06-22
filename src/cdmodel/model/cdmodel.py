from typing import Final, Literal

import torch
from lightning import pytorch as pl
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.common.data import ConversationBatch
from cdmodel.model.components.decoder import (
    DecoderCell,
    DualAttentionDecoderCell,
    FusedDualAttentionDecoderCell,
    NoneAttentionDecoderCell,
)
from cdmodel.model.components.embeddings_encoder import EmbeddingsEncoder
from cdmodel.model.components.encoder import Encoder, EncoderCell, EncoderType
from cdmodel.model.components.ist_encoder import ISTEncoder
from cdmodel.model.types import (
    AttentionActivation,
    AttentionMaskingStrategy,
    EmbeddingInputs,
    FeatureFormat,
    IstInputs,
    SpeakerInputs,
    SpeakerSexInputs,
)
from cdmodel.model.util import append_context, get_history_mask
from cdmodel.util.visualization import plot_weights


class CDModel(pl.LightningModule):
    def __init__(
        self,
        feature_names: list[str],
        emb_in: EmbeddingInputs,
        emb_proj: bool,
        emb_style: str | None,
        spk_in: SpeakerInputs,
        spk_sex_in: SpeakerSexInputs,
        enc_h_dim: int,
        enc_layers: int,
        att_mask_strategy: AttentionMaskingStrategy,
        dec_h_dim: int,
        dec_layers: int,
        lin_layers: int,
        lin_h_dim: int,
        ar_train: bool,
        ar_val: bool,
        train_primary_speaker_only: bool,
        output: FeatureFormat,
        input: list[FeatureFormat],
        enc_skip_conn: bool,
        dec_skip_conn: bool,
        att_style: (
            Literal["single"] | Literal["dual"] | Literal["fused"] | Literal["none"]
        ),
        emb_dim: int = 0,
        emb_proj_dim: int = 0,
        learn_rnn_initial_state: bool = False,
        ist: bool = False,
        ist_tokens: int = 0,
        ist_dim: int = 0,
        ist_layers: int = 0,
        ist_in: IstInputs = [],
        ist_inputs=[],
        ist_heads=1,
        ist_bidirectional: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_features: Final[int] = len(feature_names)
        self.ar_train: Final[bool] = ar_train
        self.ar_val: Final[bool] = ar_val

        self.att_style: (
            Literal["single"] | Literal["dual"] | Literal["fused"] | Literal["none"]
        ) = att_style
        if (
            att_style == "dual" or att_style == "fused"
        ) and att_mask_strategy != "both":
            raise Exception(
                "Dual and fused attention requires the 'both' attention masking strategy!"
            )
        self.att_mask_strategy: Final[AttentionMaskingStrategy] = att_mask_strategy

        self.train_primary_speaker_only: Final[bool] = train_primary_speaker_only
        self.output: Final[FeatureFormat] = output
        self.input: Final[list[FeatureFormat]] = input

        # Interaction Style Tokens
        # ==============================
        self.ist_inputs: Final[list[str]] = ist_inputs
        ist_input_dim = (
            (self.num_features * ("feature_delta_sides" in self.ist_inputs))
            + (self.num_features * ("feature_sides" in self.ist_inputs))
            + (emb_dim * ("embedding_sides" in self.ist_inputs))
            + (2 * ("sides_exchange" in self.ist_inputs))
        )
        self.enc_ist_in: Final[bool] = "encoder" in ist_in
        self.att_ist_in: Final[bool] = "attention" in ist_in
        self.dec_ist_in: Final[bool] = "decoder" in ist_in
        self.lin_ist_in: Final[bool] = "linear" in ist_in
        self.ist_dim: Final[int] = ist_dim
        self.ist_enc: nn.Module | None = None
        if ist:
            self.ist_enc = ISTEncoder(
                in_dim=ist_input_dim,
                num_tokens=ist_tokens,
                token_dim=ist_dim,
                num_layers=ist_layers,
                learn_rnn_initial_state=learn_rnn_initial_state,
                bidirectional=ist_bidirectional,
                heads=ist_heads,
            )

        # Encoder
        # ==============================
        if enc_skip_conn and att_mask_strategy != "both":
            raise Exception(
                "Encoder skip connection is only implemented for a 'both' attention masking strategy"
            )
        self.enc_emb_in: Final[bool] = "encoder" in emb_in
        self.enc_spk_in: Final[bool] = "encoder" in spk_in
        self.enc_spk_sex_in: Final[bool] = "encoder" in spk_sex_in
        self.enc_skip_conn: Final[bool] = enc_skip_conn
        enc_in_dim = (
            (self.num_features * len(self.input))
            + (2 * self.enc_spk_in)
            + (2 * self.enc_spk_sex_in)
            + (emb_proj_dim * self.enc_emb_in)
            + (ist_dim * self.enc_ist_in)
        )

        self.enc: EncoderType
        if ar_train or ar_val:
            self.enc = EncoderCell(
                input_dim=enc_in_dim,
                hidden_dim=enc_h_dim,
                num_layers=enc_layers,
                learn_rnn_initial_state=learn_rnn_initial_state,
            )
        else:
            self.enc = Encoder(
                input_dim=enc_in_dim,
                hidden_dim=enc_h_dim,
                num_layers=enc_layers,
                learn_rnn_initial_state=learn_rnn_initial_state,
            )

        # Decoder
        # ==============================
        self.att_emb_in: Final[bool] = "attention" in emb_in
        self.dec_emb_in: Final[bool] = "decoder" in emb_in
        self.lin_emb_in: Final[bool] = "linear" in emb_in
        self.att_spk_in: Final[bool] = "attention" in spk_in
        self.dec_spk_in: Final[bool] = "decoder" in spk_in
        self.att_spk_sex_in: Final[bool] = "attention" in spk_sex_in
        self.dec_spk_sex_in: Final[bool] = "decoder" in spk_sex_in
        self.lin_spk_sex_in: Final[bool] = "linear" in spk_sex_in
        self.enc_h_dim: Final[int] = enc_h_dim
        self.dec_h_dim: Final[int] = dec_h_dim
        self.dec_layers: Final[int] = dec_layers

        DecoderClass: nn.Module | None = None
        if self.att_style == "single":
            DecoderClass = DecoderCell
        elif self.att_style == "dual":
            DecoderClass = DualAttentionDecoderCell
        elif self.att_style == "fused":
            DecoderClass = FusedDualAttentionDecoderCell
        elif self.att_style == "none":
            DecoderClass = NoneAttentionDecoderCell
        else:
            raise Exception(f"Unknown attention style {self.att_style}")

        self.dec = DecoderClass(
            in_dim=enc_h_dim,
            att_ctx_dim=(
                (2 * self.att_spk_in)
                + (2 * self.att_spk_sex_in)
                + (emb_proj_dim * self.att_emb_in)
                + (ist_dim * self.att_ist_in)
            ),
            dec_ctx_dim=(
                (2 * self.dec_spk_in)
                + (2 * self.dec_spk_sex_in)
                + (emb_proj_dim * self.dec_emb_in)
                + (ist_dim * self.dec_ist_in)
            ),
            lin_ctx_dim=(
                (emb_proj_dim * self.lin_emb_in)
                + (ist_dim * self.lin_ist_in)
                + (2 * self.lin_spk_sex_in)
            ),
            h_dim=dec_h_dim,
            num_layers=dec_layers,
            lin_num_layers=lin_layers,
            lin_h_dim=lin_h_dim,
            features=feature_names,
            skip_conn=dec_skip_conn,
            learn_rnn_initial_state=learn_rnn_initial_state,
        )

        # Embeddings
        # ==============================
        self.emb_style: Final[str | None] = emb_style
        self.emb_enc = EmbeddingsEncoder(
            emb_style=emb_style,
            emb_proj=emb_proj,
            emb_dim=emb_dim,
            emb_proj_dim=emb_proj_dim,
        )
        if self.emb_enc.enabled:
            if (
                not self.enc_emb_in
                and not self.att_emb_in
                and not self.dec_emb_in
                and not self.lin_emb_in
            ):
                raise Exception(
                    "Embeddings are enabled, but they are not given as input to any component"
                )
        else:
            if self.enc_emb_in or self.att_emb_in or self.dec_emb_in or self.lin_emb_in:
                raise Exception(
                    "Embeddings are configured as model component inputs, but embeddings are disabled."
                )

    def forward(
        self,
        f: Tensor,
        ist_input: dict[int, Tensor],
        sides_lengths: dict[int, Tensor],
        conv_lengths: Tensor,
        spk_side: Tensor,
        spk_sex: Tensor,
        segment_emb: Tensor | None,
        autoregressive: bool,
    ):
        # Pull some metadata from the inputs
        (batch_size, num_steps, _), device = f.shape, f.device

        # Prepare embeddings
        emb_proj: Tensor = self.emb_enc(
            emb=segment_emb, b=batch_size, n=num_steps, device=device
        )

        # Prepare various input data
        # ==============================
        spk_side_onehot = F.one_hot(spk_side)[:, :, 1:]
        spk_is_1 = spk_side == 1
        spk_is_2 = spk_side == 2
        spk_is_1_t_arr = spk_is_1.unbind(1)

        style_emb: dict[int, Tensor] = {}

        # Prepare IST embedding
        if self.ist_enc is not None:
            ist_emb = torch.zeros(batch_size, num_steps, self.ist_dim, device=device)

            ist_1, ist_1_w = self.ist_enc(ist_input[1], sides_lengths[1])
            ist_2, ist_2_w = self.ist_enc(ist_input[2], sides_lengths[2])

            style_emb[1] = ist_1
            style_emb[2] = ist_2

            ist_emb[spk_is_1] = (
                style_emb[1]
                .type_as(ist_emb)[:, None, :]
                .expand(batch_size, num_steps, -1)[spk_is_1]
            )
            ist_emb[spk_is_2] = (
                style_emb[2]
                .type_as(ist_emb)[:, None, :]
                .expand(batch_size, num_steps, -1)[spk_is_2]
            )
        else:
            ist_emb = torch.zeros(batch_size, num_steps, 0, device=device)
            style_emb[1] = ist_emb
            style_emb[2] = ist_emb

        # Prepare encoder inputs
        # ==============================
        enc_in = append_context(
            tensors=[f, spk_side_onehot, spk_sex, emb_proj, ist_emb],
            cond=[
                True,
                self.enc_spk_in,
                self.enc_spk_sex_in,
                self.enc_emb_in,
                self.enc_ist_in,
            ],
            b=batch_size,
            n=num_steps,
            device=device,
        )
        enc_in_t_arr = enc_in.unsqueeze(2).unbind(1)

        # Prepare decoder inputs
        # ==============================
        att_ctx_t_arr = append_context(
            tensors=[spk_side_onehot, spk_sex, emb_proj, ist_emb],
            cond=[
                self.att_spk_in,
                self.att_spk_sex_in,
                self.att_emb_in,
                self.att_ist_in,
            ],
            b=batch_size,
            n=num_steps,
            device=device,
        )[:, 1:].unbind(1)
        dec_ctx_t_arr = append_context(
            tensors=[spk_side_onehot, spk_sex, emb_proj, ist_emb],
            cond=[
                self.dec_spk_in,
                self.dec_spk_sex_in,
                self.dec_emb_in,
                self.dec_ist_in,
            ],
            b=batch_size,
            n=num_steps,
            device=device,
        )[:, 1:, None].unbind(1)
        lin_ctx_t_arr = append_context(
            tensors=[spk_sex, emb_proj, ist_emb],
            cond=[self.lin_spk_sex_in, self.lin_emb_in, self.lin_ist_in],
            b=batch_size,
            n=num_steps,
            device=device,
        )[:, 1:, None].unbind(1)

        # Get state objects for the encoder and decoder
        # ==============================
        hist, enc_h = self.enc.init(input=enc_in[:, :-1], lengths=conv_lengths - 1)
        dec_h = self.dec.init(batch_size=batch_size, device=device)
        att_precomputed_keys = (
            None
            if (
                isinstance(self.enc, EncoderCell)
                or self.att_style in {"dual", "fused", "none"}
            )
            else self.dec.attention.precompute_keys(hist)
        )

        # Create the history mask.
        hist_mask_t_arr = get_history_mask(
            spk_side, self.att_mask_strategy, self.att_style
        )

        y_hat_all = torch.zeros(
            batch_size, num_steps - 1, self.num_features, device=device
        )

        # Loop through the conversation
        # ==============================
        enc_in_t: Tensor = enc_in_t_arr[0]
        att_w_arr: list[Tensor] = []
        for i in range(num_steps - 1):
            enc_h = self.enc(i=i, history=hist, h=enc_h, input=enc_in_t)
            y_hat_t, dec_h, w_t = self.dec(
                input=hist,
                h=dec_h,
                mask=hist_mask_t_arr[i],
                att_ctx=att_ctx_t_arr[i],
                dec_ctx=dec_ctx_t_arr[i],
                lin_ctx=lin_ctx_t_arr[i],
                encoder_skip=hist[:, i, None] if self.enc_skip_conn else None,
                precomputed_keys=att_precomputed_keys,
            )

            y_hat_all[:, i] = y_hat_t.squeeze(1)

            if self.att_style != "none":
                att_w_arr.append(
                    torch.concat(
                        [x.unsqueeze(1).squeeze(-1).unsqueeze(2) for x in w_t], dim=1
                    )
                )

            # Handle autoregressive training if enabled
            enc_in_t = enc_in_t_arr[i + 1]
            if autoregressive:
                enc_in_t = enc_in_t.clone()
                spk_is_primary_t = spk_is_1_t_arr[i + 1]
                enc_in_t[spk_is_primary_t, :, : self.num_features] = y_hat_t.detach()[
                    spk_is_primary_t
                ].type_as(enc_in_t)

        return (
            y_hat_all,
            torch.concat(att_w_arr, dim=2) if self.att_style != "none" else None,
            style_emb,
            (
                (
                    torch.stack([x[0] for x in hist_mask_t_arr], dim=1),
                    torch.stack([x[1] for x in hist_mask_t_arr], dim=1),
                )
                if self.att_style in {"dual", "fused"}
                else torch.stack(hist_mask_t_arr, dim=1)
            ),
        )

    def training_step(self, batch: ConversationBatch, batch_idx: int):
        batch_size: Final[int] = batch.features.shape[0]
        device = batch.features.device

        X = []
        if "feature" in self.input:
            X.append(batch.features)
        if "feature_delta" in self.input:
            X.append(batch.features_d)

        ist_input: dict[int, Tensor] = {}
        if self.ist_enc is not None:
            ist_input_arr: dict[int, list[Tensor]] = {1: [], 2: []}
            if "feature_delta_sides" in self.ist_inputs:
                ist_input_arr[1].append(batch.features_d_sides[1])
                ist_input_arr[2].append(batch.features_d_sides[2])
            if "feature_sides" in self.ist_inputs:
                ist_input_arr[1].append(batch.features_sides[1])
                ist_input_arr[2].append(batch.features_sides[2])
            if "sides_exchange" in self.ist_inputs:
                ist_input_arr[1].append(batch.sides_exchange[1])
                ist_input_arr[2].append(batch.sides_exchange[2])
            if "embedding_sides" in self.ist_inputs:
                ist_input_arr[1].append(batch.embeddings_sides[1])
                ist_input_arr[2].append(batch.embeddings_sides[2])
            ist_input[1] = torch.concat(ist_input_arr[1], -1)
            ist_input[2] = torch.concat(ist_input_arr[2], -1)

        y_hat, weights, _, masks = self(
            f=torch.concat(X, dim=-1),
            ist_input=ist_input,
            sides_lengths=batch.sides_lengths,
            conv_lengths=batch.conv_lengths,
            spk_side=batch.speaker_side,
            spk_sex=batch.speaker_sex,
            segment_emb=batch.segment_embeddings,
            autoregressive=self.ar_train,
        )

        if self.output == "feature":
            y = batch.features[:, 1:]
        elif self.output == "feature_delta":
            y = batch.features_d[:, 1:]
        else:
            raise ValueError(f"Unknown output format {self.output}")

        if self.train_primary_speaker_only:
            mask = batch.speaker_side[:, 1:] == 1
        else:
            mask = batch.speaker_side[:, 1:] != 0

        loss = F.mse_loss(y_hat[mask], y[mask])

        self.log(
            "training_loss",
            loss.detach(),
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Compute weight entropy
        if self.att_style == "single":
            mask = (
                torch.arange(max(batch.conv_lengths) - 1, device=device)[
                    None, :
                ].expand(batch_size, -1)
                < (batch.conv_lengths - 1).unsqueeze(1)
            )[:, 2:]
            entropy = (
                (-weights * weights.log()).masked_fill(~masks.unsqueeze(1), 0).sum(-1)
                / masks.sum(-1).log().unsqueeze(1)
            )[:, :, 2:][mask.unsqueeze(1)].mean()

            self.log(
                "training_entropy",
                entropy,
                on_epoch=True,
                on_step=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        elif self.att_style == "dual" or self.att_style == "fused":
            mask = (
                torch.arange(max(batch.conv_lengths) - 1, device=device)[
                    None, :
                ].expand(batch_size, -1)
                < (batch.conv_lengths - 1).unsqueeze(1)
            )[:, 2:]

            mask0_sum = masks[0].sum(-1)
            mask1_sum = masks[1].sum(-1)
            entropy1 = (
                (-weights[:, 0] * weights[:, 0].log()).masked_fill(~masks[0], 0).sum(-1)
                / mask0_sum.log()
            )[mask0_sum > 1].mean()
            entropy2 = (
                (-weights[:, 1] * weights[:, 1].log()).masked_fill(~masks[1], 0).sum(-1)
                / mask1_sum.log()
            )[mask1_sum > 1].mean()

            self.log(
                "training_entropy_partner",
                entropy1,
                on_epoch=True,
                on_step=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "training_entropy_self",
                entropy2,
                on_epoch=True,
                on_step=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        elif self.att_sytle == "none":
            pass
        else:
            raise Exception(f"Unknown attention style {self.att_style}!")
        return loss

    def validation_step(self, batch: ConversationBatch, batch_idx: int):
        batch_size: Final[int] = batch.features.shape[0]
        device = batch.features.device

        X = []
        if "feature" in self.input:
            X.append(batch.features)
        if "feature_delta" in self.input:
            X.append(batch.features_d)

        ist_input: dict[int, Tensor] = {}
        if self.ist_enc is not None:
            ist_input_arr: dict[int, list[Tensor]] = {1: [], 2: []}
            if "feature_delta_sides" in self.ist_inputs:
                ist_input_arr[1].append(batch.features_d_sides[1])
                ist_input_arr[2].append(batch.features_d_sides[2])
            if "feature_sides" in self.ist_inputs:
                ist_input_arr[1].append(batch.features_sides[1])
                ist_input_arr[2].append(batch.features_sides[2])
            if "sides_exchange" in self.ist_inputs:
                ist_input_arr[1].append(batch.sides_exchange[1])
                ist_input_arr[2].append(batch.sides_exchange[2])
            if "embedding_sides" in self.ist_inputs:
                ist_input_arr[1].append(batch.embeddings_sides[1])
                ist_input_arr[2].append(batch.embeddings_sides[2])
            ist_input[1] = torch.concat(ist_input_arr[1], -1)
            ist_input[2] = torch.concat(ist_input_arr[2], -1)

        y_hat, weights, _, masks = self(
            f=torch.concat(X, dim=-1),
            ist_input=ist_input,
            sides_lengths=batch.sides_lengths,
            conv_lengths=batch.conv_lengths,
            spk_side=batch.speaker_side,
            spk_sex=batch.speaker_sex,
            segment_emb=batch.segment_embeddings,
            autoregressive=self.ar_val,
        )

        if self.output == "feature":
            y = batch.features[:, 1:]
        elif self.output == "feature_delta":
            y = batch.features_d[:, 1:]
        else:
            raise ValueError(f"Unknown output format {self.output}")

        if self.train_primary_speaker_only:
            mask = batch.speaker_side[:, 1:] == 1
        else:
            mask = batch.speaker_side[:, 1:] != 0

        loss = F.mse_loss(y_hat[mask], y[mask])

        self.log(
            "validation_loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Compute weight entropy
        if self.att_style == "single":
            mask = (
                torch.arange(max(batch.conv_lengths) - 1, device=device)[
                    None, :
                ].expand(batch_size, -1)
                < (batch.conv_lengths - 1).unsqueeze(1)
            )[:, 2:]
            entropy = (
                (-weights * weights.log()).masked_fill(~masks.unsqueeze(1), 0).sum(-1)
                / masks.sum(-1).log().unsqueeze(1)
            )[:, :, 2:][mask.unsqueeze(1)].mean()

            self.log(
                "validation_entropy",
                entropy,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                batch_size=batch_size,
            )
        elif self.att_style == "dual" or self.att_style == "fused":
            mask = (
                torch.arange(max(batch.conv_lengths) - 1, device=device)[
                    None, :
                ].expand(batch_size, -1)
                < (batch.conv_lengths - 1).unsqueeze(1)
            )[:, 2:]

            mask0_sum = masks[0].sum(-1)
            mask1_sum = masks[1].sum(-1)
            entropy1 = (
                (-weights[:, 0] * weights[:, 0].log()).masked_fill(~masks[0], 0).sum(-1)
                / mask0_sum.log()
            )[mask0_sum > 1].mean()
            entropy2 = (
                (-weights[:, 1] * weights[:, 1].log()).masked_fill(~masks[1], 0).sum(-1)
                / mask1_sum.log()
            )[mask1_sum > 1].mean()

            self.log(
                "validation_entropy_partner",
                entropy1,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "validation_entropy_self",
                entropy2,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                batch_size=batch_size,
            )
        elif self.att_sytle == "none":
            pass
        else:
            raise Exception(f"Unknown attention style {self.att_style}!")

        if (
            self.global_rank == 0
            and batch_idx == 0
            and self.logger is not None
            and self.att_style != "none"
        ):
            conv_length: int = int(batch.conv_lengths[0].item())

            w = weights[0, :, : conv_length - 1, :conv_length].cpu().numpy()
            spk_side = batch.speaker_side[0, :conv_length].cpu().numpy()

            plot_weights(
                weights=w,
                masking_strategy=self.att_mask_strategy,
                att_style=self.att_style,
                spk_side=spk_side,
                spk=1,
                title="weights_1",
                logger=self.logger.experiment,
                global_step=self.global_step,
            )
            plot_weights(
                weights=w,
                masking_strategy=self.att_mask_strategy,
                att_style=self.att_style,
                spk_side=spk_side,
                spk=2,
                title="weights_2",
                logger=self.logger.experiment,
                global_step=self.global_step,
            )
            plot_weights(
                weights=w,
                masking_strategy=self.att_mask_strategy,
                att_style=self.att_style,
                spk_side=spk_side,
                spk=None,
                title="weights_both",
                logger=self.logger.experiment,
                global_step=self.global_step,
            )

        return loss

    def predict_step(self, batch: ConversationBatch, batch_idx: int):
        X = []
        if "feature" in self.input:
            X.append(batch.features)
        if "feature_delta" in self.input:
            X.append(batch.features_d)

        ist_input: dict[int, Tensor] = {}
        if self.ist_enc is not None:
            ist_input_arr: dict[int, list[Tensor]] = {1: [], 2: []}
            if "feature_delta_sides" in self.ist_inputs:
                ist_input_arr[1].append(batch.features_d_sides[1])
                ist_input_arr[2].append(batch.features_d_sides[2])
            if "feature_sides" in self.ist_inputs:
                ist_input_arr[1].append(batch.features_sides[1])
                ist_input_arr[2].append(batch.features_sides[2])
            if "sides_exchange" in self.ist_inputs:
                ist_input_arr[1].append(batch.sides_exchange[1])
                ist_input_arr[2].append(batch.sides_exchange[2])
            if "embedding_sides" in self.ist_inputs:
                ist_input_arr[1].append(batch.embeddings_sides[1])
                ist_input_arr[2].append(batch.embeddings_sides[2])
            ist_input[1] = torch.concat(ist_input_arr[1], -1)
            ist_input[2] = torch.concat(ist_input_arr[2], -1)

        y_hat, weights, style_emb, hist_mask = self(
            f=torch.concat(X, dim=-1),
            ist_input=ist_input,
            sides_lengths=batch.sides_lengths,
            conv_lengths=batch.conv_lengths,
            spk_side=batch.speaker_side,
            spk_sex=batch.speaker_sex,
            segment_emb=batch.segment_embeddings,
            autoregressive=self.ar_val,
        )

        return batch, y_hat, weights, style_emb, hist_mask

    def on_before_optimizer_step(self, optimizer):
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            for tag, param in self.named_parameters():
                self.logger.experiment.add_histogram(
                    f"grad/{tag}", param.grad.data.cpu().numpy(), self.global_step
                )
                self.logger.experiment.add_histogram(
                    f"weight/{tag}", param.detach().cpu().numpy(), self.global_step
                )

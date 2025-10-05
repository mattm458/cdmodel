from typing import Final

import torch
from lightning import pytorch as pl
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.common.data import ConversationBatch
from cdmodel.model.components.decoder import DecoderCell
from cdmodel.model.components.embeddings_encoder import EmbeddingsEncoder
from cdmodel.model.components.encoder import Encoder, EncoderCell, EncoderType
from cdmodel.model.components.ist_encoder import ISTEncoder
from cdmodel.model.types import (
    AttentionMaskingStrategy,
    EmbeddingInputs,
    FeatureFormat,
    IstInputs,
    SpeakerInputs,
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
        dec_skip_conn: bool,
        emb_dim: int = 0,
        emb_proj_dim: int = 0,
        learn_rnn_initial_state: bool = False,
        ist: bool = False,
        ist_tokens: int = 0,
        ist_dim: int = 0,
        ist_layers: int = 0,
        ist_h_dim: int = 0,
        ist_in: IstInputs = [],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_features: Final[int] = len(feature_names)
        self.ar_train: Final[bool] = ar_train
        self.ar_val: Final[bool] = ar_val

        self.att_mask_strategy: Final[AttentionMaskingStrategy] = att_mask_strategy

        self.train_primary_speaker_only: Final[bool] = train_primary_speaker_only
        self.output: Final[FeatureFormat] = output
        self.input: Final[list[FeatureFormat]] = input

        # Interaction Style Tokens
        # ==============================
        self.att_ist_in: Final[bool] = "attention" in ist_in
        self.dec_ist_in: Final[bool] = "decoder" in ist_in
        self.lin_ist_in: Final[bool] = "linear" in ist_in
        self.ist_dim: Final[int] = ist_dim
        self.ist_enc: nn.Module | None = None
        if ist:
            self.ist_enc = ISTEncoder(
                in_dim=self.num_features,
                num_tokens=ist_tokens,
                token_dim=ist_dim,
                hidden_dim=ist_h_dim,
                num_layers=ist_layers,
                learn_rnn_initial_state=learn_rnn_initial_state,
            )

        # Encoder
        # ==============================
        self.enc_emb_in: Final[bool] = "encoder" in emb_in
        self.enc_spk_in: Final[bool] = "encoder" in spk_in
        enc_in_dim = (
            (self.num_features * len(self.input))
            + (2 * self.enc_spk_in)
            + (emb_proj_dim * self.enc_emb_in)
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
        self.enc_h_dim: Final[int] = enc_h_dim
        self.dec_h_dim: Final[int] = dec_h_dim
        self.dec_layers: Final[int] = dec_layers
        self.dec = DecoderCell(
            in_dim=enc_h_dim,
            att_ctx_dim=(2 * self.att_spk_in)
            + (emb_proj_dim * self.att_emb_in)
            + (ist_dim * self.att_ist_in),
            dec_ctx_dim=(2 * self.dec_spk_in)
            + (emb_proj_dim * self.dec_emb_in)
            + (ist_dim * self.dec_ist_in),
            lin_ctx_dim=(emb_proj_dim * self.lin_emb_in) + (ist_dim * self.lin_ist_in),
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
        f_d_sides: dict[int, Tensor],
        sides_lengths: dict[int, Tensor],
        conv_lengths: Tensor,
        spk_side: Tensor,
        segment_emb: Tensor | None,
        autoregressive: bool,
        return_h: bool = False,
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

        # Prepare IST embedding
        if self.ist_enc is not None:
            ist_emb = torch.zeros(batch_size, num_steps, self.ist_dim, device=device)

            ist_1, ist_1_w = self.ist_enc(f_d_sides[1], sides_lengths[1])
            ist_2, ist_2_w = self.ist_enc(f_d_sides[2], sides_lengths[2])

            ist_emb[spk_is_1] = ist_1.type_as(ist_emb)[:, None, :].expand(
                -1, num_steps, -1
            )[spk_is_1]
            ist_emb[spk_is_2] = ist_2.type_as(ist_emb)[:, None, :].expand(
                -1, num_steps, -1
            )[spk_is_2]
        else:
            ist_emb = torch.zeros(batch_size, num_steps, 0, device=device)

        # Prepare encoder inputs
        # ==============================
        enc_in = append_context(
            tensors=[f, spk_side_onehot, emb_proj],
            cond=[True, self.enc_spk_in, self.enc_emb_in],
            b=batch_size,
            n=num_steps,
            device=f.device,
        )
        enc_in_t_arr = enc_in.unsqueeze(2).unbind(1)

        # Prepare decoder inputs
        # ==============================
        att_ctx_t_arr = append_context(
            tensors=[spk_side_onehot, emb_proj, ist_emb],
            cond=[self.att_spk_in, self.att_emb_in, self.att_ist_in],
            b=batch_size,
            n=num_steps,
            device=f.device,
        )[:, 1:].unbind(1)
        dec_ctx_t_arr = append_context(
            tensors=[spk_side_onehot, emb_proj, ist_emb],
            cond=[self.dec_spk_in, self.dec_emb_in, self.dec_ist_in],
            b=batch_size,
            n=num_steps,
            device=f.device,
        )[:, 1:, None].unbind(1)
        lin_ctx_t_arr = append_context(
            tensors=[emb_proj, ist_emb],
            cond=[self.lin_emb_in, self.lin_ist_in],
            b=batch_size,
            n=num_steps,
            device=f.device,
        )[:, 1:, None].unbind(1)

        # Get state objects for the encoder and decoder
        # ==============================
        hist, enc_h = self.enc.init(input=enc_in[:, :-1], lengths=conv_lengths - 1)
        precomputed_keys: Tensor | None = None
        if not autoregressive:
            precomputed_keys = self.dec.attention.precompute_keys(hist)
        dec_h = self.dec.init(batch_size=batch_size, device=device)

        # Create the history mask.
        hist_mask_t_arr = get_history_mask(spk_side, self.att_mask_strategy).unbind(1)

        y_hat_all = torch.zeros(
            batch_size, num_steps - 1, self.num_features, device=device
        )
        att_h_all: Tensor | None = None
        att_w_all: Tensor | None = None
        dec_h_all: Tensor | None = None
        if return_h:
            att_h_all = torch.zeros(
                batch_size, num_steps - 1, self.enc_h_dim, device=device
            )
            att_w_all = torch.zeros(
                batch_size, num_steps - 1, num_steps - 1, 1, device=device
            )
            dec_h_all = torch.zeros(
                batch_size,
                num_steps - 1,
                self.dec_h_dim * self.dec_layers,
                device=device,
            )

        # Loop through the conversation
        # ==============================
        enc_in_t: Tensor = enc_in_t_arr[0]
        for i in range(num_steps - 1):
            enc_h = self.enc(i=i, history=hist, h=enc_h, input=enc_in_t)
            y_hat_t, dec_h, att_t, w_t = self.dec(
                input=hist,
                h=dec_h,
                mask=hist_mask_t_arr[i],
                att_ctx=att_ctx_t_arr[i],
                dec_ctx=dec_ctx_t_arr[i],
                lin_ctx=lin_ctx_t_arr[i],
                precomputed_keys=precomputed_keys,
            )

            y_hat_all[:, i] = y_hat_t.squeeze(1)
            if att_w_all is not None:
                att_w_all[:, i] = w_t
            if att_h_all is not None:
                att_h_all[:, i] = att_t.detach().squeeze(1)
            if dec_h_all is not None:
                dec_h_all[:, i] = dec_h.detach().swapaxes(0, 1).reshape(batch_size, -1)

            # Handle autoregressive training if enabled
            enc_in_t = enc_in_t_arr[i + 1]
            if autoregressive:
                spk_is_primary_t = spk_is_1_t_arr[i + 1]
                enc_in_t[spk_is_primary_t, :, : self.num_features] = y_hat_t.detach()[
                    spk_is_primary_t
                ].type_as(enc_in_t)

        return y_hat_all, att_w_all, att_h_all, dec_h_all

    def training_step(self, batch: ConversationBatch, batch_idx: int):
        batch_size: Final[int] = batch.features.shape[0]

        X = []
        if "feature" in self.input:
            X.append(batch.features)
        if "feature_delta" in self.input:
            X.append(batch.features_d)

        y_hat, _, _, _ = self(
            f=torch.concat(X, dim=-1),
            f_d_sides=batch.features_d_sides,
            sides_lengths=batch.sides_lengths,
            conv_lengths=batch.conv_lengths,
            spk_side=batch.speaker_side,
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

        return loss

    def validation_step(self, batch: ConversationBatch, batch_idx: int):
        batch_size: Final[int] = batch.features.shape[0]

        X = []
        if "feature" in self.input:
            X.append(batch.features)
        if "feature_delta" in self.input:
            X.append(batch.features_d)

        y_hat, weights, _, _ = self(
            f=torch.concat(X, dim=-1),
            f_d_sides=batch.features_d_sides,
            sides_lengths=batch.sides_lengths,
            conv_lengths=batch.conv_lengths,
            spk_side=batch.speaker_side,
            segment_emb=batch.segment_embeddings,
            autoregressive=self.ar_val,
            return_h=True,
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
            loss.detach(),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        if self.global_rank == 0 and batch_idx == 0 and self.logger is not None:
            conv_length: int = int(batch.conv_lengths[0].item())
            w = weights[0, : conv_length - 1, :conv_length].squeeze().cpu().numpy()
            spk_side = batch.speaker_side[0, :conv_length].cpu().numpy()

            plot_weights(
                weights=w,
                spk_side=spk_side,
                spk=1,
                title="weights_1",
                logger=self.logger.experiment,
                global_step=self.global_step,
            )
            plot_weights(
                weights=w,
                spk_side=spk_side,
                spk=2,
                title="weights_2",
                logger=self.logger.experiment,
                global_step=self.global_step,
            )
            plot_weights(
                weights=w,
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

        y_hat, weights, att, dec = self(
            f=torch.concat(X, dim=-1),
            f_d_sides=batch.features_d_sides,
            sides_lengths=batch.sides_lengths,
            conv_lengths=batch.conv_lengths,
            spk_side=batch.speaker_side,
            segment_emb=batch.segment_embeddings,
            autoregressive=self.ar_val,
            return_h=True,
        )

        return batch, y_hat, weights, att, dec

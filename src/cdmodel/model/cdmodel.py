from typing import Final, Literal

import torch
from lightning import pytorch as pl
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.common.data import ConversationBatch
from cdmodel.model.components.decoder import DecoderCell
from cdmodel.model.components.embeddings_encoder import EmbeddingsEncoder
from cdmodel.model.components.encoder import Encoder, EncoderCell, EncoderType
from cdmodel.util.visualization import plot_weights

AttentionMaskingStrategy = Literal["partner"] | Literal["both"]


def _append_context(tensors: list[Tensor | None], cond: list[bool], b: int, n: int):
    lst = [t for (t, c) in zip(tensors, cond) if c and t is not None]
    if len(lst) == 0:
        return torch.zeros(b, n, 0)
    return torch.concat(lst, -1)


class CDModel(pl.LightningModule):
    def __init__(
        self,
        feature_names: list[str],
        emb_style: str | None,
        emb_proj: bool,
        enc_h_dim: int,
        enc_layers: int,
        enc_spk_in: bool,
        enc_emb_in: bool,
        att_emb_in: bool,
        att_spk_in: bool,
        att_mask_strategy: AttentionMaskingStrategy,
        dec_h_dim: int,
        dec_layers: int,
        dec_emb_in: bool,
        dec_spk_in: bool,
        lin_layers: int,
        lin_h_dim: int,
        lin_emb_in: bool,
        ar_train: bool,
        ar_val: bool,
        train_primary_speaker_only: bool,
        emb_dim: int = 0,
        emb_proj_dim: int = 0,
    ):
        super().__init__()

        self.num_features: Final[int] = len(feature_names)
        self.ar_train: Final[bool] = ar_train
        self.ar_val: Final[bool] = ar_val

        self.att_mask_strategy: Final[AttentionMaskingStrategy] = att_mask_strategy

        self.emb_style: Final[str | None] = emb_style
        self.enc_emb_in: Final[bool] = enc_emb_in
        self.att_emb_in: Final[bool] = att_emb_in
        self.att_spk_in: Final[bool] = att_spk_in
        self.dec_emb_in: Final[bool] = dec_emb_in
        self.lin_emb_in: Final[bool] = lin_emb_in

        self.train_primary_speaker_only: Final[bool] = train_primary_speaker_only

        # Embeddings
        # ==============================
        self.emb_enc = EmbeddingsEncoder(
            emb_style=emb_style,
            emb_proj=emb_proj,
            emb_dim=emb_dim,
            emb_proj_dim=emb_proj_dim,
        )
        if self.emb_enc.enabled:
            if not enc_emb_in and not att_emb_in and not dec_emb_in and not lin_emb_in:
                raise Exception(
                    "Embeddings are enabled, but they are not given as input to any component"
                )
        else:
            if enc_emb_in or att_emb_in or dec_emb_in or lin_emb_in:
                raise Exception(
                    "Embeddings are configured as model component inputs, but embeddings are disabled."
                )

        # Encoder
        # ==============================
        self.enc_spk_in: Final[bool] = enc_spk_in
        enc_in_dim = self.num_features + (2 * enc_spk_in) + (emb_proj_dim * enc_emb_in)

        self.enc: EncoderType
        if ar_train or ar_val:
            self.enc = EncoderCell(
                input_dim=enc_in_dim, hidden_dim=enc_h_dim, num_layers=enc_layers
            )
        else:
            self.enc = Encoder(
                input_dim=enc_in_dim, hidden_dim=enc_h_dim, num_layers=enc_layers
            )

        # Decoder
        # ==============================
        self.dec_spk_in: Final[bool] = dec_spk_in
        self.dec = DecoderCell(
            in_dim=enc_h_dim,
            att_ctx_dim=(2 * att_spk_in) + (emb_proj_dim * att_emb_in),
            dec_ctx_dim=(2 * dec_spk_in) + (emb_proj_dim * dec_emb_in),
            lin_ctx_dim=emb_proj_dim * lin_emb_in,
            h_dim=dec_h_dim,
            num_layers=dec_layers,
            lin_num_layers=lin_layers,
            lin_h_dim=lin_h_dim,
            features=feature_names,
        )

    def forward(
        self,
        f: Tensor,
        conv_lengths: Tensor,
        spk_rank: Tensor,
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
        spk_rank_one_hot = F.one_hot(spk_rank)[:, :, 1:]  # Speaker rank one-hot encoded
        spk_is_primary_t_arr = (spk_rank == 1).unbind(1)  # Identify primary speakers

        # Prepare encoder inputs
        # ==============================
        enc_in = _append_context(
            tensors=[f, spk_rank_one_hot, emb_proj],
            cond=[True, self.enc_spk_in, self.enc_emb_in],
            b=batch_size,
            n=num_steps,
        )

        # Prepare decoder inputs
        # ==============================
        att_ctx_t_arr = _append_context(
            tensors=[spk_rank_one_hot[:, 1:], emb_proj[:, 1:]],
            cond=[self.att_spk_in, self.att_emb_in],
            b=batch_size,
            n=num_steps,
        ).unbind(1)
        dec_ctx_t_arr = _append_context(
            tensors=[spk_rank_one_hot[:, 1:], emb_proj[:, 1:]],
            cond=[self.dec_spk_in, self.dec_emb_in],
            b=batch_size,
            n=num_steps,
        ).split(1, 1)
        lin_ctx_t_arr = _append_context(
            tensors=[emb_proj[:, 1:]], cond=[self.lin_emb_in], b=batch_size, n=num_steps
        ).split(1, 1)

        # Get state objects for the encoder and decoder
        # ==============================
        hist, enc_h = self.enc.init(input=enc_in, lengths=conv_lengths)
        dec_h = self.dec.init(batch_size=batch_size, device=device)

        # Create the history mask.
        # The history mask tensor has the following dimensions:
        #   [batch, conv_timestep, hist_timestep]
        # At each conv_timestep, the tensor contains an attention
        # mask that hides irrelevant historical turns from the
        # attention mechanism.
        # TODO: Test this
        spk_rank_hist = spk_rank[:, None, :-1]
        spk_rank_pred = spk_rank[:, 1:, None]
        if self.att_mask_strategy == "partner":
            spk_rank_cond = (spk_rank_hist != spk_rank_pred) & (spk_rank_hist != 0)
        elif self.att_mask_strategy == "both":
            spk_rank_cond = spk_rank_hist.repeat(1, num_steps - 1, 1) != 0
        else:
            raise ValueError(
                f"Unknown attention mask strategy {self.att_mask_strategy}"
            )
        hist_mask_t_arr = spk_rank_cond.tril().unbind(1)

        y_hat = torch.zeros(batch_size, num_steps - 1, self.num_features, device=device)
        w = torch.zeros(batch_size, num_steps - 1, num_steps - 1, 1, device=device)

        # Loop through the conversation
        # ==============================
        enc_in_t: Tensor = enc_in[:, 0, None]
        for i in range(num_steps - 1):
            enc_h = self.enc(i=i, history=hist, h=enc_h, input=enc_in_t)
            y_hat_t, dec_h, w_t = self.dec(
                input=hist,
                h=dec_h,
                mask=hist_mask_t_arr[i],
                att_ctx=att_ctx_t_arr[i],
                dec_ctx=dec_ctx_t_arr[i],
                lin_ctx=lin_ctx_t_arr[i],
            )

            y_hat[:, i] = y_hat_t.squeeze(1)
            w[:, i] = w_t

            # Handle autoregressive training if enabled
            enc_in_t = enc_in[:, i + 1, None]
            if autoregressive:
                spk_is_primary_t = spk_is_primary_t_arr[i + 1]
                enc_in_t[spk_is_primary_t, :, : self.num_features] = y_hat_t.detach()[
                    spk_is_primary_t
                ].type(enc_in_t.dtype)

        return y_hat, w

    def training_step(self, batch: ConversationBatch, batch_idx: int):
        batch_size: Final[int] = batch.features.shape[0]

        y_hat, weights = self(
            f=batch.features,
            conv_lengths=batch.conv_lengths,
            spk_rank=batch.speaker_rank,
            segment_emb=batch.segment_embeddings,
            autoregressive=self.ar_train,
        )
        y = batch.features[:, 1:]

        if self.train_primary_speaker_only:
            mask = batch.speaker_rank[:, 1:] == 1
        else:
            mask = batch.speaker_rank[:, 1:] != 0

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

        y_hat, weights = self(
            f=batch.features,
            conv_lengths=batch.conv_lengths,
            spk_rank=batch.speaker_rank,
            segment_emb=batch.segment_embeddings,
            autoregressive=self.ar_val,
        )
        y = batch.features[:, 1:]

        if self.train_primary_speaker_only:
            mask = batch.speaker_rank[:, 1:] == 1
        else:
            mask = batch.speaker_rank[:, 1:] != 0

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
            sr = batch.speaker_rank[0, :conv_length].cpu().numpy()
            self.logger.experiment.add_figure(
                "weights_1",
                plot_weights(weights=w, spk_rank=sr, spk=1),
                global_step=self.global_step,
            )
            self.logger.experiment.add_figure(
                "weights_2",
                plot_weights(weights=w, spk_rank=sr, spk=2),
                global_step=self.global_step,
            )

        return loss

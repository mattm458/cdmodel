from typing import Final

import torch
from lightning import pytorch as pl
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.common.data import ConversationBatch
from cdmodel.model.components.decoder import DecoderCell
from cdmodel.model.components.encoder import Encoder, EncoderCell, EncoderType
from cdmodel.util.visualization import plot_weights


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
        dec_h_dim: int,
        dec_layers: int,
        dec_emb_in: bool,
        dec_spk_in: bool,
        lin_num_layers: int,
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

        self.emb_style: Final[str | None] = emb_style
        self.enc_emb_in: Final[bool] = enc_emb_in
        self.att_emb_in: Final[bool] = att_emb_in
        self.att_spk_in: Final[bool] = att_spk_in
        self.dec_emb_in: Final[bool] = dec_emb_in
        self.lin_emb_in: Final[bool] = lin_emb_in

        self.train_primary_speaker_only: Final[bool] = train_primary_speaker_only

        # Embeddings
        # ==============================
        self.use_emb: Final[bool] = emb_style is not None

        self.emb_proj: nn.Module | None = None
        if self.use_emb:
            if emb_dim == 0:
                raise ValueError(
                    "emb_dim must be specified when embeddings are enabled."
                )
            if not enc_emb_in and not att_emb_in and not dec_emb_in and not lin_emb_in:
                raise Exception(
                    "Embeddings are enabled, but they are not given as input to any component"
                )

            self.emb_proj = nn.Identity()
            if emb_proj:
                self.emb_proj = nn.Sequential(
                    nn.Linear(emb_dim, emb_proj_dim), nn.Tanh()
                )
            else:
                if emb_proj_dim != 0 and emb_proj_dim != emb_dim:
                    raise ValueError(
                        "If emb_proj is False, emb_proj_dim must equal emb_dim."
                    )
                emb_proj_dim = emb_proj_dim or emb_dim
        else:
            if emb_proj:
                raise ValueError("emb_proj is True, but embeddings are disabled.")
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
                input_dim=enc_in_dim,
                hidden_dim=enc_h_dim,
                num_layers=enc_layers,
            )
        else:
            self.enc = Encoder(
                input_dim=enc_in_dim,
                hidden_dim=enc_h_dim,
                num_layers=enc_layers,
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
            lin_num_layers=lin_num_layers,
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

        if self.use_emb and segment_emb is None:
            raise Exception(
                "Embeddings are active, but no embeddings were given as input"
            )

        # Prepare embeddings
        emb_proj: Tensor | None = None
        if self.use_emb and self.emb_proj:
            emb_proj = self.emb_proj(segment_emb)

        # Prepare various input data
        # ==============================
        spk_rank_one_hot = F.one_hot(spk_rank)[:, :, 1:]  # Speaker rank one-hot encoded
        spk_is_primary = spk_rank == 1  # Identify primary speakers

        # Prepare encoder inputs
        # ==============================
        enc_in_arr: list[Tensor] = [f]
        if self.enc_spk_in:
            enc_in_arr.append(spk_rank_one_hot)
        if self.enc_emb_in and emb_proj is not None:
            enc_in_arr.append(emb_proj)
        enc_in = torch.cat(enc_in_arr, -1)

        # Prepare decoder inputs
        # ==============================
        att_ctx_arr: list[Tensor] = []
        att_ctx: Tensor = torch.zeros(batch_size, num_steps, 0)
        if self.att_spk_in:
            att_ctx_arr.append(spk_rank_one_hot[:, 1:])
        if self.att_emb_in and emb_proj is not None:
            att_ctx_arr.append(emb_proj[:, 1:])
        if len(att_ctx_arr):
            att_ctx = torch.concat(att_ctx_arr, -1)

        dec_ctx_arr: list[Tensor] = []
        dec_ctx: Tensor = torch.zeros(batch_size, num_steps, 0)
        if self.dec_spk_in:
            dec_ctx_arr.append(spk_rank_one_hot[:, 1:])
        if self.dec_emb_in and emb_proj is not None:
            dec_ctx_arr.append(emb_proj[:, 1:])
        if len(dec_ctx_arr):
            dec_ctx = torch.concat(dec_ctx_arr, -1)

        lin_ctx_arr: list[Tensor] = []
        lin_ctx: Tensor = torch.zeros(batch_size, num_steps, 0)
        if self.lin_emb_in and emb_proj is not None:
            lin_ctx_arr.append(emb_proj[:, 1:])
        if len(lin_ctx_arr):
            lin_ctx = torch.concat(lin_ctx_arr, -1)

        # Get state objects for the encoder and decoder
        # ==============================
        enc_state = self.enc.init(
            input=enc_in, lengths=conv_lengths, speaker_rank=spk_rank
        )
        dec_state = self.dec.init(batch_size=batch_size, device=device)

        y_hat_arr: list[Tensor] = []
        w_arr: list[Tensor] = []

        enc_in_t: Tensor = enc_in[:, 0]

        # Loop through the conversation
        # ==============================
        for i in range(num_steps - 1):
            hist, hist_mask = self.enc(i=i, state=enc_state, input=enc_in_t)

            y_hat_t, w_t = self.dec(
                state=dec_state,
                input=hist,
                mask=hist_mask,
                att_ctx=att_ctx[:, None, i],
                dec_ctx=dec_ctx[:, None, i],
                lin_ctx=lin_ctx[:, None, i],
            )

            y_hat_arr.append(y_hat_t)
            w_arr.append(w_t)

            # Handle autoregressive training if enabled
            enc_in_t = enc_in[:, i + 1]
            if autoregressive:
                spk_is_primary_t = spk_is_primary[:, i + 1]
                enc_in_t[spk_is_primary_t, : self.num_features] = (
                    y_hat_t.squeeze(1)[spk_is_primary_t].detach().type(enc_in_t.dtype)
                )

        y_hat = torch.concat(y_hat_arr, dim=1)

        # Batch, prediction timesteps, history, 1
        # TODO: Move this to decoder?
        w = torch.stack(
            [F.pad(x, (0, 0, 0, f.shape[1] - x.shape[1] - 1)) for x in w_arr],
            1,
        )

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

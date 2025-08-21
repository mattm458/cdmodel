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

        self.feature_names: Final[list[str]] = feature_names
        self.autoregressive_training: Final[bool] = ar_train
        self.autoregressive_validation: Final[bool] = ar_val

        self.train_primary_speaker_only: Final[bool] = train_primary_speaker_only

        # Embeddings
        # ==============================
        self.use_emb: Final[bool] = emb_style is not None
        self.emb_style: Final[str | None] = emb_style
        self.enc_emb_in: Final[bool] = enc_emb_in
        self.att_emb_in: Final[bool] = att_emb_in
        self.att_spk_in: Final[bool] = att_spk_in
        self.dec_emb_in: Final[bool] = dec_emb_in
        self.lin_emb_in: Final[bool] = lin_emb_in

        self.emb_proj: nn.Module | None = None
        if self.use_emb:
            if emb_dim == 0:
                raise ValueError(
                    "emb_dim must be specified when embeddings are enabled."
                )
            if not (enc_emb_in and att_emb_in and dec_emb_in and lin_emb_in):
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
        enc_in_dim = (
            len(feature_names)
            + (2 if enc_spk_in else 0)
            + (emb_proj_dim if enc_emb_in else 0)
        )

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

        att_ctx_dim = (2 if att_spk_in else 0) + (emb_proj_dim if att_emb_in else 0)
        dec_ctx_dim = (2 if dec_spk_in else 0) + (emb_proj_dim if dec_emb_in else 0)
        lin_ctx_dim = emb_proj_dim if lin_emb_in else 0

        self.dec = DecoderCell(
            in_dim=enc_h_dim,
            att_ctx_dim=att_ctx_dim,
            ctx_dim=dec_ctx_dim,
            lin_ctx_dim=lin_ctx_dim,
            h_dim=dec_h_dim,
            num_layers=dec_layers,
            lin_num_layers=lin_num_layers,
            features=feature_names,
        )

    def forward(
        self,
        features: Tensor,
        conv_lengths: Tensor,
        speaker_designation: Tensor,
        segment_emb: Tensor | None,
        autoregressive: bool,
    ):
        # Pull some metadata from the inputs
        (batch_size, num_steps, _), device = features.shape, features.device

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
        # Speaker designations, one-hot encoded for model input
        speaker_designation_one_hot = F.one_hot(speaker_designation)[:, :, 1:]

        # History masks for timestep-level prediction
        # The dimensions of the tensors below are:
        #
        #   batch x prediction timesteps x dialogue history timesteps
        speaker_designation_timesteps = speaker_designation[:, None, :-1]
        history_mask = (
            speaker_designation_timesteps != speaker_designation[:, 1:].unsqueeze(2)
        ) & (speaker_designation_timesteps != 0)

        # Prepare encoder inputs
        # ==============================
        encoder_in_arr: list[Tensor] = [features]
        if self.enc_spk_in:
            encoder_in_arr.append(speaker_designation_one_hot)
        if self.enc_emb_in and emb_proj is not None:
            encoder_in_arr.append(emb_proj)
        enc_in = torch.cat(encoder_in_arr, -1)

        # Prepare decoder inputs
        # ==============================
        additional_att_in_arr: list[Tensor] = []
        att_ctx: Tensor | None = None
        if self.att_spk_in:
            additional_att_in_arr.append(speaker_designation_one_hot[:, 1:])
        if self.att_emb_in and emb_proj is not None:
            additional_att_in_arr.append(emb_proj[:, 1:])
        if len(additional_att_in_arr):
            att_ctx = torch.concat(additional_att_in_arr, -1)

        dec_ctx_arr: list[Tensor] = []
        dec_ctx: Tensor | None = None
        if self.dec_spk_in:
            dec_ctx_arr.append(speaker_designation_one_hot[:, 1:])
        if self.dec_emb_in and emb_proj is not None:
            dec_ctx_arr.append(emb_proj[:, 1:])
        if len(dec_ctx_arr):
            dec_ctx = torch.concat(dec_ctx_arr, -1)

        lin_ctx_arr: list[Tensor] = []
        lin_ctx: Tensor | None = None
        if self.lin_emb_in and emb_proj is not None:
            lin_ctx_arr.append(emb_proj[:, 1:])
        if len(lin_ctx_arr):
            lin_ctx = torch.concat(lin_ctx_arr, -1)

        # Get state objects for the encoder and decoder
        # ==============================
        enc_state = self.enc.init(input=enc_in, lengths=conv_lengths)
        dec_state = self.dec.init(batch_size=batch_size, device=device)

        decoded_timesteps: list[Tensor] = []
        weights_timesteps: list[Tensor] = []

        enc_in_t: Tensor = enc_in[:, 0]

        # Loop through the conversation
        # ==============================
        for i in range(num_steps - 1):
            # Encode the current inputs and retrieve the conversation history so far
            history = self.enc(i=i, state=enc_state, input=enc_in_t)

            # Decode output features
            dec_t, w_t = self.dec(
                state=dec_state,
                input=history,
                mask=history_mask[:, i, : i + 1],
                att_ctx=(att_ctx[:, None, i] if att_ctx is not None else None),
                dec_ctx=(dec_ctx[:, None, i] if dec_ctx is not None else None),
                lin_ctx=(lin_ctx[:, None, i] if lin_ctx is not None else None),
            )

            decoded_timesteps.append(dec_t)
            weights_timesteps.append(w_t)

            # Handle autoregressive training if enabled
            enc_in_t = enc_in[:, i + 1]
            if autoregressive:
                speaker_is_primary_t = speaker_designation[:, i + 1] == 1
                enc_in_t[speaker_is_primary_t, : len(self.feature_names)] = (
                    dec_t.squeeze(1)[speaker_is_primary_t].detach().type(enc_in_t.dtype)
                )

        decoded = torch.concat(decoded_timesteps, dim=1)

        # Batch, prediction timesteps, history, 1
        # TODO: Move this to decoder?
        weights = torch.stack(
            [
                F.pad(x, (0, 0, 0, features.shape[1] - x.shape[1] - 1))
                for x in weights_timesteps
            ],
            1,
        )

        return decoded, weights

    def training_step(self, batch: ConversationBatch, batch_idx: int):
        batch_size: Final[int] = batch.features.shape[0]

        y_hat, weights = self(
            features=batch.features,
            conv_lengths=batch.conv_lengths,
            speaker_designation=batch.speaker_designation,
            segment_emb=batch.segment_embeddings,
            autoregressive=self.autoregressive_training,
        )
        y = batch.features[:, 1:]

        if self.train_primary_speaker_only:
            mask = batch.speaker_designation[:, 1:] == 1
        else:
            mask = batch.speaker_designation[:, 1:] != 0

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
            features=batch.features,
            conv_lengths=batch.conv_lengths,
            speaker_designation=batch.speaker_designation,
            segment_emb=batch.segment_embeddings,
            autoregressive=self.autoregressive_validation,
        )
        y = batch.features[:, 1:]

        if self.train_primary_speaker_only:
            mask = batch.speaker_designation[:, 1:] == 1
        else:
            mask = batch.speaker_designation[:, 1:] != 0

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
            sd = batch.speaker_designation[0, :conv_length].cpu().numpy()
            self.logger.experiment.add_figure(
                "weights_1",
                plot_weights(weights=w, speaker_designation=sd, speaker=1),
                global_step=self.global_step,
            )
            self.logger.experiment.add_figure(
                "weights_2",
                plot_weights(weights=w, speaker_designation=sd, speaker=2),
                global_step=self.global_step,
            )

        return loss

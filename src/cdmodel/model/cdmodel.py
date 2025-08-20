from typing import Final

import torch
from lightning import pytorch as pl
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.common.data import ConversationBatch
from cdmodel.model.components.decoder import DecoderCell
from cdmodel.model.components.encoder import Encoder, EncoderCell
from cdmodel.util.visualization import plot_weights


class CDModel(pl.LightningModule):
    def __init__(
        self,
        features: list[str],
        embeddings_style: str | None,
        embeddings_linear: bool,
        encoder_hidden_dim: int,
        encoder_num_layers: int,
        encoder_speaker_in: bool,
        encoder_embeddings_in: bool,
        att_embeddings_in: bool,
        decoder_hidden_dim: int,
        decoder_num_layers: int,
        decoder_embeddings_in: bool,
        decoder_speaker_in: bool,
        decoder_num_linear_layers: int,
        decoder_linear_embeddings_in: bool,
        embeddings_in_dim: int = 0,
        embeddings_dim: int = 0,
    ):
        super().__init__()
        # Embeddings
        # ==============================
        if embeddings_style is not None:
            if embeddings_in_dim == 0:
                raise Exception("embeddings_in_dim must be specified ")
        if embeddings_style is None and (
            encoder_embeddings_in
            or att_embeddings_in
            or decoder_embeddings_in
            or decoder_linear_embeddings_in
        ):
            raise Exception("Embeddings must be active to use as input!")
        self.embeddings_style: Final[str | None] = embeddings_style
        self.encoder_embeddings_in: Final[bool] = encoder_embeddings_in
        self.att_embeddings_in: Final[bool] = att_embeddings_in
        self.decoder_embeddings_in: Final[bool] = decoder_embeddings_in
        self.decoder_linear_embeddings_in: Final[bool] = decoder_linear_embeddings_in

        self.embeddings_linear = None
        if embeddings_linear:
            self.embeddings_linear = nn.Sequential(
                nn.Linear(embeddings_in_dim, embeddings_dim), nn.Tanh()
            )
        else:
            if embeddings_dim == 0:
                embeddings_dim = embeddings_in_dim
            elif embeddings_dim != 0 and embeddings_dim != embeddings_in_dim:
                raise Exception(
                    "A model with embeddings with no an embedding transformation layer must have embeddings_in_dim = embeddings_dim!"
                )

        # Encoder
        # ==============================
        self.encoder_speaker_in: Final[bool] = encoder_speaker_in
        encoder_input_dim = (
            len(features)
            + (2 if encoder_speaker_in else 0)
            + (embeddings_dim if encoder_embeddings_in else 0)
        )

        self.encoder = Encoder(
            input_dim=encoder_input_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
        )

        # Decoder
        # ==============================
        self.decoder_speaker_in: Final[bool] = decoder_speaker_in
        additional_att_dim = embeddings_dim if att_embeddings_in else 0
        additional_decoder_dim = (2 if decoder_speaker_in else 0) + (
            embeddings_dim if decoder_embeddings_in else 0
        )
        additional_decoder_linear_dim = (
            embeddings_dim if decoder_linear_embeddings_in else 0
        )

        self.decoder = DecoderCell(
            input_dim=encoder_hidden_dim,
            additional_att_dim=additional_att_dim,
            additional_decoder_dim=additional_decoder_dim,
            additional_decoder_linear_dim=additional_decoder_linear_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            num_linear_layers=decoder_num_linear_layers,
            features=features,
        )

    def forward(
        self,
        features: Tensor,
        conv_lengths: Tensor,
        speaker_designation: Tensor,
        segment_embeddings: Tensor | None,
    ):
        # Pull some metadata from the inputs
        batch_size, num_steps, _ = features.shape

        # Prepare embeddings
        embeddings_encoded = None
        if segment_embeddings is not None:
            if self.embeddings_linear is not None:
                embeddings_encoded = self.embeddings_linear(segment_embeddings)
            else:
                embeddings_encoded = segment_embeddings

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
        if self.encoder_speaker_in:
            encoder_in_arr.append(speaker_designation_one_hot)
        if self.encoder_embeddings_in and embeddings_encoded is not None:
            encoder_in_arr.append(embeddings_encoded)
        encoder_in = torch.cat(encoder_in_arr, -1)

        # Prepare decoder inputs
        # ==============================
        additional_att_in_arr: list[Tensor] = []
        additional_att_in: Tensor | None = None
        if self.att_embeddings_in and embeddings_encoded is not None:
            additional_att_in_arr.append(embeddings_encoded[:, 1:])
        if len(additional_att_in_arr):
            additional_att_in = torch.concat(additional_att_in_arr, -1)

        additional_decoder_in_arr: list[Tensor] = []
        additional_decoder_in: Tensor | None = None
        if self.decoder_speaker_in:
            additional_decoder_in_arr.append(speaker_designation_one_hot)
        if self.decoder_embeddings_in and embeddings_encoded is not None:
            additional_decoder_in_arr.append(embeddings_encoded[:, 1:])
        if len(additional_decoder_in_arr):
            additional_decoder_in = torch.concat(additional_decoder_in_arr, -1)

        additional_decoder_linear_in_arr: list[Tensor] = []
        additional_decoder_linear_in: Tensor | None = None
        if self.decoder_linear_embeddings_in and embeddings_encoded is not None:
            additional_decoder_linear_in_arr.append(embeddings_encoded[:, 1:])
        if len(additional_decoder_linear_in_arr):
            additional_decoder_linear_in = torch.concat(
                additional_decoder_linear_in_arr, -1
            )

        # Get state objects for the encoder and decoder
        # ==============================
        encoder_state = self.encoder.initialize(input=encoder_in, lengths=conv_lengths)
        decoder_state = self.decoder.initialize(
            batch_size=batch_size, device=features.device
        )

        decoded_timesteps: list[Tensor] = []
        weights_timesteps: list[Tensor] = []

        # Loop through the conversation
        # ==============================
        for i in range(num_steps - 1):
            # Encode the current inputs and retrieve the conversation history so far
            history = self.encoder(i=i, state=encoder_state, input=encoder_in[:, i, :])

            # Decode output features
            decoded_timestep, weights_timestep = self.decoder(
                state=decoder_state,
                input=history,
                mask=history_mask[:, i, : i + 1],
                additional_att_in=(
                    additional_att_in[:, None, i]
                    if additional_att_in is not None
                    else None
                ),
                additional_decoder_in=(
                    additional_decoder_in[:, None, i]
                    if additional_decoder_in is not None
                    else None
                ),
                additional_decoder_linear_in=(
                    additional_decoder_linear_in[:, None, i]
                    if additional_decoder_linear_in is not None
                    else None
                ),
            )

            decoded_timesteps.append(decoded_timestep)
            weights_timesteps.append(weights_timestep)

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
            segment_embeddings=batch.segment_embeddings,
        )
        y = batch.features[:, 1:]

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
            segment_embeddings=batch.segment_embeddings,
        )
        y = batch.features[:, 1:]

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

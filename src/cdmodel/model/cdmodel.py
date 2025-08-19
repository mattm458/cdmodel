from typing import Final

import torch
from lightning import pytorch as pl
from torch import Tensor
from torch.nn import functional as F

from cdmodel.common.data import ConversationBatch
from cdmodel.model.components.decoder import DecoderCell
from cdmodel.model.components.encoder import Encoder, EncoderCell
from cdmodel.util.visualization import plot_weights


class CDModel(pl.LightningModule):
    def __init__(
        self,
        features: list[str],
        encoder_hidden_size: int,
        encoder_num_layers: int,
        encoder_speaker_in: bool,
    ):
        super().__init__()

        self.encoder_speaker_in: Final[bool] = encoder_speaker_in
        encoder_input_size: Final[int] = len(features) + (
            2 if encoder_speaker_in else 0
        )

        self.encoder = Encoder(
            input_size=encoder_input_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
        )

        self.decoder = DecoderCell(
            input_size=32, hidden_size=32, num_layers=2, features=features
        )

    def forward(
        self, features: Tensor, conv_lengths: Tensor, speaker_designation: Tensor
    ):
        # Pull some metadata from the inputs
        batch_size, num_steps, _ = features.shape

        encoder_in_arr: list[Tensor] = [features]
        if self.encoder_speaker_in:
            encoder_in_arr.append(F.one_hot(speaker_designation)[:, :, 1:])

        encoder_in = torch.cat(encoder_in_arr, -1)

        # Get state objects for the encoder and decoder
        encoder_state = self.encoder.initialize(input=encoder_in, lengths=conv_lengths)
        decoder_state = self.decoder.initialize(
            batch_size=batch_size, device=features.device
        )

        decoded_timesteps: list[Tensor] = []
        weights_timesteps: list[Tensor] = []

        # Loop through the conversation
        for i in range(num_steps - 1):
            # Encode the current inputs and retrieve the conversation history so far
            history = self.encoder(i=i, state=encoder_state, input=encoder_in[:, i, :])

            # NEW: Who is the opposite partner in the dialogue history?
            speaker_pred = speaker_designation[:, i + 1, None]
            speaker_history = speaker_designation[:, : i + 1]
            mask = (speaker_history != speaker_pred) & (speaker_history != 0)

            # Decode output features
            decoded_timestep, weights_timestep = self.decoder(
                state=decoder_state,
                input=history,
                mask=mask,
            )

            decoded_timesteps.append(decoded_timestep)
            weights_timesteps.append(weights_timestep)

        decoded = torch.concat(decoded_timesteps, dim=1)

        # Batch, prediction timesteps, history, 1
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
            batch.features, batch.conv_lengths, batch.speaker_designation
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
            batch.features, batch.conv_lengths, batch.speaker_designation
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

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001)

from typing import Literal

import torch
from torch import Tensor

from cdmodel.model.types import AttentionMaskingStrategy


def append_context(
    tensors: list[Tensor | None], cond: list[bool], b: int, n: int, device
):
    lst = [t for (t, c) in zip(tensors, cond) if c and t is not None]
    if len(lst) == 0:
        return torch.zeros(b, n, 0, device=device)
    return torch.concat(lst, -1)


def get_history_mask(
    speaker_side: Tensor,
    att_mask_strategy: AttentionMaskingStrategy,
    att_style: Literal["single"] | Literal["dual"] | Literal["fused"] | Literal["none"],
):
    """
    Build a causal attention mask over a history of dyadic conversation turns.
    Constructing the mask is based on two factors:
        - Who is speaking at each historical turn versus who is speaking at the
          upcoming turn.
        - The masking strategy employed by the model.

    Inputs
     ------
    speaker_side : Tensor[batch, timesteps]
        Integer speaker side per timestep according to the following convention:
        - Values of 0 are padding
        - Values of 1 or 2 indicate sides of a conversation

    att_mask_strategy : {"partner", "both"}
        - "partner": at each output timestep t, only allow attention to historical
          turns spoken by the other speaker.
        - "both": at each output timestep t, allow attention to any historical turn,
          regardless of speaker. Padding is ignored

    Returns
    -------
    Tensor[b, t_out, t_hist]
        A Boolean tensor where a True value at [b, t_out, t_hist] means that,
        when producing output at timestep t_out+1, you may attend to historical
        timestep t_hist.

        Notes:
          - The mask is causal. Only history up to the output step is allowed.
          - The mask is given as a diagonal. It is intended to be split and used one
            t_out at a time.
    """

    # history_turns: [B, 1, 0 to (T-1)]
    history = speaker_side[:, None, :-1]
    batch_size = history.shape[0]
    device = history.device

    if att_style == "single":
        if att_mask_strategy == "partner":
            # output_turns: [B, 1 to T, 1]
            output = speaker_side[:, 1:, None]

            # Allow only:
            #   - (history_turns != 0):             Non-padding history
            #   - (history_turns != output_turns):  History spoken by the other side
            spk_side_mask = (history != output) & (history != 0)
        elif att_mask_strategy == "self":
            # output_turns: [B, 1 to T, 1]
            output = speaker_side[:, 1:, None]

            # Allow only:
            #   - (history_turns != 0):             Non-padding history
            #   - (history_turns != output_turns):  History spoken by us
            spk_side_mask = (history == output) & (history != 0)
        elif att_mask_strategy == "both":
            num_timesteps = speaker_side.shape[1]

            # Allow all non-padding history
            spk_side_mask = (history != 0).expand(-1, (num_timesteps - 1), -1)

        else:
            raise ValueError(f"Unknown attention mask strategy {att_mask_strategy}")

        return spk_side_mask.tril().unbind(1)
    elif att_style == "none":
        return (
            torch.eye(n=speaker_side.shape[1] - 1, device=device)
            .bool()[None, :]
            .expand(batch_size, -1, -1)
            .unbind(1)
        )
    elif att_style == "dual" or att_style == "fused":
        # Partner mask
        # output_turns: [B, 1 to T, 1]
        output = speaker_side[:, 1:, None]

        partner_spk_side_mask = ((history != output) & (history != 0)).tril().unbind(1)
        self_spk_side_mask = ((history == output) & (history != 0)).tril().unbind(1)

        return tuple(zip(partner_spk_side_mask, self_spk_side_mask))
    else:
        raise ValueError(f"Unknown attention style {att_style}")

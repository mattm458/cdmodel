import torch
from torch import Tensor

from cdmodel.model.types import AttentionMaskingStrategy


@torch.compile
def append_context(
    tensors: list[Tensor | None], cond: list[bool], b: int, n: int, device
):
    lst = [t for (t, c) in zip(tensors, cond) if c and t is not None]
    if len(lst) == 0:
        return torch.zeros(b, n, 0, device=device)
    return torch.concat(lst, -1)


@torch.compile
def get_history_mask(speaker_side: Tensor, att_mask_strategy: AttentionMaskingStrategy):
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

    if att_mask_strategy == "partner":
        # output_turns: [B, 1 to T, 1]
        output = speaker_side[:, 1:, None]

        # Allow only:
        #   - (history_turns != 0):             Non-padding history
        #   - (history_turns != output_turns):  History spoken by the other side
        spk_side_mask = (history != output) & (history != 0)
    elif att_mask_strategy == "both":
        num_timesteps = speaker_side.shape[1]

        # Allow all non-padding history
        spk_side_mask = (history != 0).expand(-1, (num_timesteps - 1), -1)
    else:
        raise ValueError(f"Unknown attention mask strategy {att_mask_strategy}")

    # Make the mask causal with tril()
    return spk_side_mask.tril()

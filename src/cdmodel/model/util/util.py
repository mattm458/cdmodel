from typing import Final, Literal, Optional

import torch
from torch import Generator, Tensor
from torch.nn import functional as F

from cdmodel.common.consts import SPEAKER_ROLE_AGENT_IDX, SPEAKER_ROLE_PARTNER_IDX


def one_hot_drop_0(tensor: Tensor, num_classes: int = -1) -> Tensor:
    return F.one_hot(tensor, num_classes=num_classes)[:, :, 1:]


def lengths_to_mask(lengths: Tensor, max_size: int) -> Tensor:
    mask = torch.arange(max_size, device=lengths.device)
    mask = mask.unsqueeze(0).repeat(len(lengths), 1)
    mask = mask >= lengths.unsqueeze(1)
    mask = mask.unsqueeze(2)
    return mask


def timestep_split(tensor: Tensor) -> list[Tensor]:
    return [x.squeeze(1) for x in torch.split(tensor, 1, dim=1)]


def get_role_identity_idx(
    speaker_identity_idx: Tensor,
    role_assignment: Literal["agent_first", "agent_second", "random"],
    zero_pad: bool,
    generator: Optional[Generator] = None,
) -> tuple[Tensor, Tensor]:
    """
    Given a tensor of speaker identity indices in a segmented conversation,
    assign roles ("agent" or "partner") to the speakers.

    Roles are assigned based on one of three strategies:

    * `agent_first`: The agent is the first person to speak in the conversation, and
      their partner is the second to speak.
    * `agent_second`: The agent is the second person to speak in the conversation, and
      their partner is the first to speak.
    * `random`: Agent and partner roles are assigned randomly.

    For random assignment, a PyTorch Generator object is required. This is to ensure
    that random numbers are assigned on the same device as the output tensor, and
    that situations requiring deterministic output (for example, random role selection
    during validation) can be set up in advance.

    Parameters
    ----------
    speaker_identity_idx : Tensor
        A tensor of batched segmented speaker identity indices. It must have the
        dimensions (batch, segments).
    role_assignment : Literal["agent_first", "agent_second", "random"]
        The method used to assign speaker roles.
    zero_pad : bool
        Whether the `speaker_identity_idx` tensor is zero-padded. If it is, the first
        index along the segment dimension is 0, which is not a valid speaker identity.
    generator : Optional[Generator], optional
        A PyTorch Generator for random numbers. Required if `role_assignment` is
        `random`. By default `None`.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple containing two tensors: the speaker identity index of the agents in each
        batch, and the speaker identity index of the partners in each batch.

    Raises
    ------
    Exception
        Raised if the role assignment strategy is `random` but no Generator was given
        as an argument. Alternatively, if speaker_identity_idx is zero-padded but the
        zero_pad argument is given as False.
    NotImplementedError
        Raised if an unknown role assignment strategy is given as an argument.
    """
    if torch.any(speaker_identity_idx[:, 0] == 0) and not zero_pad:
        raise Exception(
            "speaker_identity_idx appears to be zero-padded, but 'zero_pad' is False!"
        )

    if zero_pad:
        speaker_identity_idx = speaker_identity_idx[:, int(zero_pad) :]

    batch_size: Final[int] = speaker_identity_idx.shape[0]
    speaker_ids: list[Tensor] = [
        x.unique() for x in speaker_identity_idx.split(1, dim=0)
    ]
    speaker_ids = [x[x != 0] for x in speaker_ids]

    first_speaker_idx = speaker_identity_idx[:, 0]
    second_speaker_idx = torch.concat(
        [
            batch_speaker_ids[batch_speaker_ids != partner_id]
            for batch_speaker_ids, partner_id in zip(
                speaker_ids, first_speaker_idx.split(1)
            )
        ]
    )

    if role_assignment == "agent_first":
        agent_identity_idx = first_speaker_idx
        partner_identity_idx = second_speaker_idx
    elif role_assignment == "agent_second":
        agent_identity_idx = second_speaker_idx
        partner_identity_idx = first_speaker_idx
    elif role_assignment == "random":
        if generator is None:
            raise Exception(
                "If using random role assignment, a Generator object must be given"
            )

        partner_first = (
            torch.rand(batch_size, generator=generator, device=generator.device) <= 0.5
        )

        agent_identity_idx = first_speaker_idx.clone()
        agent_identity_idx[partner_first] = second_speaker_idx[partner_first]

        partner_identity_idx = second_speaker_idx.clone()
        partner_identity_idx[partner_first] = first_speaker_idx[partner_first]

    else:
        raise NotImplementedError(
            f'Role assignment method "{role_assignment}" not supported'
        )

    if torch.any(agent_identity_idx == partner_identity_idx):
        raise Exception("Debug: An agent and partner were given the same role")

    return agent_identity_idx, partner_identity_idx


def get_speaker_role_idx(
    speaker_identity_idx: Tensor,
    agent_identity_idx: Tensor,
    partner_identity_idx: Tensor,
) -> Tensor:
    """
    Generate a tensor of speaker role indices from a tensor of speaker identity indices,
    plus separate tensors indicating which speaker identities are assuming the role of agent
    and which speaker identities are assuming the role of partner.

    As an example, assume the speaker identity indices contain the following batched segments:

    ```py
    [
        [5, 6, 5, 6, 5, 0],
        [6, 7, 6, 7, 0, 0],
        [3, 4, 3, 4, 3, 3]
    ]
    ```

    If the agent identities are `[5, 6, 4]` and the partner identities are `[6, 7, 3]`,
    then the output of this function will be the following tensor:

    ```py
    [
        [2, 1, 2, 1, 2, 0],
        [2, 1, 2, 1, 0, 0],
        [1, 2, 1, 2, 1, 1]
    ]
    ```

    This is because the agent role index is hardcoded at 2, and the speaker role index at 1.
    Values of 0 are ignored, since they are used for padding.


    Parameters
    ----------
    speaker_identity_idx : Tensor
        A tensor of batched segmented speaker identity indices. It must have the
        dimensions (batch, segments).
    agent_identity_idx : Tensor
        The speaker identity indidices of the agents in each batch.
    partner_identity_idx : Tensor
        The speaker identity indidices of the partners in each batch.

    Returns
    -------
    Tensor
        A tensor of batched segmented speaker role indices with the dimensions (batch, segments).
    """
    is_agent = speaker_identity_idx == agent_identity_idx.unsqueeze(1)
    is_partner = speaker_identity_idx == partner_identity_idx.unsqueeze(1)

    speaker_role_idx = torch.zeros_like(speaker_identity_idx)
    speaker_role_idx[is_agent] = SPEAKER_ROLE_AGENT_IDX
    speaker_role_idx[is_partner] = SPEAKER_ROLE_PARTNER_IDX

    return speaker_role_idx

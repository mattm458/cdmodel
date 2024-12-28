import json
from os import path
from random import Random
from typing import Final

import torch
from torch import Tensor
from torch.functional import F
from torch.utils.data import Dataset

from cdmodel.common import ConversationData
from cdmodel.common.role_assignment import (
    AnalysisRole,
    DialogueSystemRole,
    Role,
    RoleAssignmentStrategy,
    RoleType,
    assign_speaker_roles,
)


def _load_segment_data(dataset_dir: str, conv_id: int) -> dict:
    with open(path.join(dataset_dir, "segments", f"{conv_id}.json")) as infile:
        return json.load(infile)


def _load_embeddings(dataset_dir: str, conv_id: int) -> tuple[Tensor, Tensor]:
    return torch.load(
        path.join(dataset_dir, "embeddings", f"{conv_id}-embeddings.pt"),
        weights_only=True,
    ), torch.load(
        path.join(dataset_dir, "embeddings", f"{conv_id}-lengths.pt"),
        weights_only=True,
    )


class ConversationDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        segment_features: list[str],
        zero_pad: bool,
        role_type: RoleType,
        role_assignment_strategy: RoleAssignmentStrategy,
        conv_ids: list[int],
        speaker_ids: dict[int, int],
        deterministic: bool = True,
    ):
        super().__init__()

        self.dataset_dir: Final[str] = dataset_dir
        self.conv_ids: Final[list[int]] = conv_ids
        self.segment_features: Final[list[str]] = segment_features
        self.zero_pad: Final[bool] = zero_pad
        self.speaker_ids: Final[dict[int, int]] = speaker_ids
        self.role_type: Final[RoleType] = role_type
        self.role_assignment_strategy: Final[RoleAssignmentStrategy] = (
            role_assignment_strategy
        )

        self.deterministic: Final[bool] = deterministic
        self.random = Random()

    def __len__(self) -> int:
        return len(self.conv_ids)

    def __getitem__(self, i: int) -> ConversationData:
        conv_id: Final[int] = self.conv_ids[i]

        # Load conversation data from disk
        conv_data: Final[dict] = _load_segment_data(self.dataset_dir, conv_id)
        embeddings, embeddings_turn_len = _load_embeddings(self.dataset_dir, conv_id)

        segment_features: Tensor = torch.tensor(
            [conv_data[feature] for feature in self.segment_features]
        ).swapaxes(0, 1)
        segment_features_delta = segment_features.diff(
            dim=0, prepend=torch.zeros(1, segment_features.shape[1])
        )

        speaker_id: list[int] = conv_data["speaker_id"]
        speaker_id_idx: Tensor = torch.tensor(
            [self.speaker_ids[x] for x in conv_data["speaker_id"]],
            dtype=torch.long,
        )

        # Establish speaker roles
        if self.deterministic:
            self.random.seed(i)
        speaker_role, role_speaker_assignment = assign_speaker_roles(
            speaker_ids=speaker_id,
            role_type=self.role_type,
            role_assignment_strategy=self.role_assignment_strategy,
            random=self.random,
        )
        speaker_role_idx = torch.tensor([x.value for x in speaker_role])
        role_speaker_assignment_idx = {
            k: self.speaker_ids[v] for k, v in role_speaker_assignment.items()
        }

        segment_features_delta_sides: dict[Role, Tensor] = {}
        segment_features_delta_sides_len: dict[Role, list[int]] = {}
        match self.role_type:
            case RoleType.DialogueSystem:
                segment_features_delta_sides[DialogueSystemRole.partner] = (
                    segment_features_delta[
                        speaker_role_idx == DialogueSystemRole.partner.value
                    ].unsqueeze(0)
                )
                segment_features_delta_sides_len[DialogueSystemRole.partner] = [
                    segment_features_delta[
                        speaker_role_idx == DialogueSystemRole.partner.value
                    ].shape[0]
                ]

                segment_features_delta_sides[DialogueSystemRole.agent] = (
                    segment_features_delta[
                        speaker_role_idx == DialogueSystemRole.agent.value
                    ].unsqueeze(0)
                )
                segment_features_delta_sides_len[DialogueSystemRole.agent] = [
                    segment_features_delta[
                        speaker_role_idx == DialogueSystemRole.agent.value
                    ].shape[0]
                ]
            case RoleType.Analysis:
                segment_features_delta_sides[AnalysisRole.a] = segment_features_delta[
                    speaker_role_idx == AnalysisRole.a.value
                ].unsqueeze(0)
                segment_features_delta_sides_len[AnalysisRole.a] = [
                    segment_features_delta[
                        speaker_role_idx == AnalysisRole.a.value
                    ].shape[0]
                ]

                segment_features_delta_sides[AnalysisRole.b] = segment_features_delta[
                    speaker_role_idx == AnalysisRole.b.value
                ].unsqueeze(0)
                segment_features_delta_sides_len[AnalysisRole.b] = [
                    segment_features_delta[
                        speaker_role_idx == AnalysisRole.b.value
                    ].shape[0]
                ]
            case _:
                raise Exception("Oh no")

        if self.zero_pad:
            segment_features = F.pad(segment_features, (0, 0, 1, 0))
            segment_features_delta = F.pad(segment_features_delta, (0, 0, 1, 0))
            embeddings = F.pad(embeddings, (0, 0, 0, 0, 1, 0))
            embeddings_turn_len = F.pad(embeddings_turn_len, (1, 0), value=1)
            speaker_id = [0] + speaker_id
            speaker_id_idx = F.pad(speaker_id_idx, (1, 0))
            speaker_role = [None] + speaker_role
            speaker_role_idx = F.pad(speaker_role_idx, (1, 0))

        return ConversationData(
            conv_id=[conv_id],
            segment_features=segment_features.unsqueeze(0),
            segment_features_delta=segment_features_delta.unsqueeze(0),
            segment_features_delta_sides=segment_features_delta_sides,
            segment_features_delta_sides_len=segment_features_delta_sides_len,
            embeddings=embeddings,
            embeddings_segment_len=embeddings_turn_len,
            num_segments=[segment_features.shape[0]],
            speaker_id=[speaker_id],
            speaker_id_idx=speaker_id_idx.unsqueeze(0),
            # TODO: Fix this type
            speaker_role=[speaker_role],
            speaker_role_idx=speaker_role_idx.unsqueeze(0),
            role_speaker_assignment=[role_speaker_assignment],
            role_speaker_assignment_idx=[role_speaker_assignment_idx],
        )

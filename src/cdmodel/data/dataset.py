import json
from os import path
from typing import Final

import torch
from torch import Tensor
from torch.functional import F
from torch.utils.data import Dataset

from cdmodel.common import ConversationData
from cdmodel.common.role_assignment import (
    BothPredictionRole,
    AgentPartnerPredictionRole,
    Role,
    RoleAssignmentStrategy,
    PredictionType,
    assign_speaker_roles,
)


def load_segment_data(dataset_dir: str, conv_id: int) -> dict:
    with open(path.join(dataset_dir, "segments", f"{conv_id}.json")) as infile:
        return json.load(infile)


def load_embeddings(dataset_dir: str, conv_id: int, embeddings_dir: str) -> Tensor:
    return torch.load(
        path.join(dataset_dir, embeddings_dir, f"{conv_id}-embeddings.pt"),
        weights_only=True,
    )


def load_embeddings_len(dataset_dir: str, conv_id: int, embeddings_dir: str) -> Tensor:
    return torch.load(
        path.join(dataset_dir, embeddings_dir, f"{conv_id}-lengths.pt"),
        weights_only=True,
    )


class ConversationDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        feature_names: list[str],
        zero_pad: bool,
        role_type: PredictionType,
        role_assignment_strategy: RoleAssignmentStrategy,
        conv_ids: list[int],
        speaker_ids: dict[int, int],
        embeddings_type: str | None,
    ):
        super().__init__()

        if (
            role_type == PredictionType.Both
            and role_assignment_strategy != RoleAssignmentStrategy.first
        ):
            raise Exception(
                "The 'Both' prediction type can only be used with the 'first' role assignment strategy"
            )

        self.dataset_dir: Final[str] = dataset_dir
        self.conv_ids: Final[list[int]] = conv_ids
        self.feature_names: Final[list[str]] = feature_names
        self.zero_pad: Final[bool] = zero_pad
        self.speaker_ids: Final[dict[int, int]] = speaker_ids
        self.role_type: Final[PredictionType] = role_type
        self.role_assignment_strategy: Final[RoleAssignmentStrategy] = (
            role_assignment_strategy
        )
        self.embeddings_type: Final[str | None] = embeddings_type

    def __len__(self) -> int:
        # If we use the 'both' role assignment strategy, the Dataset will
        # present itself as being twice its true size. This is so the Dataset
        # can return two instances of each conversation: once where the speaking
        # role is assigned to the first speaker, and once where the speaking role
        # is assigned to the second speaker.
        if self.role_assignment_strategy == RoleAssignmentStrategy.both:
            return len(self.conv_ids) * 2
        else:
            return len(self.conv_ids)

    def __getitem__(self, i: int) -> ConversationData:
        role_assignment_strategy: RoleAssignmentStrategy = self.role_assignment_strategy

        # For the 'both' role assignment strategy, the dataset presents itself as
        # 2x larger than its true size. Because of this, one of two outcomes is possible:
        if role_assignment_strategy == RoleAssignmentStrategy.both:
            # 1. The caller is asking for an instance in the first half of available
            # instances. If so, we'll use the 'first' role assignment strategy.
            if i < len(self.conv_ids):
                role_assignment_strategy = RoleAssignmentStrategy.first
            # 2. The caller is asking for an instance in the second half of available
            # instances. If so, we'll use the 'second' role assignment strategy.
            else:
                i %= len(self.conv_ids)
                role_assignment_strategy = RoleAssignmentStrategy.second

        conv_id: Final[int] = self.conv_ids[i]

        # Load conversation data from disk
        conv_data: Final[dict] = load_segment_data(self.dataset_dir, conv_id)

        word_embeddings = None
        word_embeddings_len = None
        segment_embeddings = None

        if self.embeddings_type == "glove":
            word_embeddings = load_embeddings(self.dataset_dir, conv_id, "glove")
            word_embeddings_len = load_embeddings_len(
                self.dataset_dir, conv_id, "glove"
            )
        elif self.embeddings_type == "roberta":
            segment_embeddings = load_embeddings(self.dataset_dir, conv_id, "roberta")

        segment_features: Tensor = torch.tensor(
            [conv_data[feature] for feature in self.feature_names]
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
        speaker_role, role_speaker_assignment = assign_speaker_roles(
            speaker_ids=speaker_id,
            prediction_type=self.role_type,
            role_assignment_strategy=role_assignment_strategy,
        )
        speaker_role_idx = torch.tensor(
            [x.value if x is not None else 0 for x in speaker_role]
        )
        role_speaker_assignment_idx = {
            k: self.speaker_ids[v] for k, v in role_speaker_assignment.items()
        }

        transcript = conv_data["transcript"]

        # Assemble conversation side data
        segment_features_sides: dict[Role, Tensor] = {}
        segment_features_delta_sides: dict[Role, Tensor] = {}
        segment_features_sides_len: dict[Role, list[int]] = {}
        if self.role_type == PredictionType.AgentPartner:
            segment_features_sides[AgentPartnerPredictionRole.partner] = (
                segment_features[
                    speaker_role_idx == AgentPartnerPredictionRole.partner.value
                ].unsqueeze(0)
            )
            segment_features_delta_sides[AgentPartnerPredictionRole.partner] = (
                segment_features_delta[
                    speaker_role_idx == AgentPartnerPredictionRole.partner.value
                ].unsqueeze(0)
            )
            segment_features_sides_len[AgentPartnerPredictionRole.partner] = [
                segment_features_delta[
                    speaker_role_idx == AgentPartnerPredictionRole.partner.value
                ].shape[0]
            ]

            segment_features_sides[AgentPartnerPredictionRole.agent] = segment_features[
                speaker_role_idx == AgentPartnerPredictionRole.agent.value
            ].unsqueeze(0)
            segment_features_delta_sides[AgentPartnerPredictionRole.agent] = (
                segment_features_delta[
                    speaker_role_idx == AgentPartnerPredictionRole.agent.value
                ].unsqueeze(0)
            )
            segment_features_sides_len[AgentPartnerPredictionRole.agent] = [
                segment_features_delta[
                    speaker_role_idx == AgentPartnerPredictionRole.agent.value
                ].shape[0]
            ]
        elif self.role_type == PredictionType.Both:
            segment_features_sides[BothPredictionRole.a] = segment_features[
                speaker_role_idx == BothPredictionRole.a.value
            ].unsqueeze(0)
            segment_features_delta_sides[BothPredictionRole.a] = segment_features_delta[
                speaker_role_idx == BothPredictionRole.a.value
            ].unsqueeze(0)
            segment_features_sides_len[BothPredictionRole.a] = [
                segment_features_delta[
                    speaker_role_idx == BothPredictionRole.a.value
                ].shape[0]
            ]

            segment_features_sides[BothPredictionRole.b] = segment_features[
                speaker_role_idx == BothPredictionRole.b.value
            ].unsqueeze(0)
            segment_features_delta_sides[BothPredictionRole.b] = segment_features_delta[
                speaker_role_idx == BothPredictionRole.b.value
            ].unsqueeze(0)
            segment_features_sides_len[BothPredictionRole.b] = [
                segment_features_delta[
                    speaker_role_idx == BothPredictionRole.b.value
                ].shape[0]
            ]
        else:
            raise Exception("Oh no")

        # If zero-padding is on, pad all of the relevant tensors
        if self.zero_pad:
            segment_features = F.pad(segment_features, (0, 0, 1, 0))
            segment_features_delta = F.pad(segment_features_delta, (0, 0, 1, 0))

            if word_embeddings is not None:
                word_embeddings = F.pad(word_embeddings, (0, 0, 0, 0, 1, 0))
            if word_embeddings_len is not None:
                word_embeddings_len = F.pad(word_embeddings_len, (1, 0), value=1)
            if segment_embeddings is not None:
                segment_embeddings = F.pad(segment_embeddings, (0, 0, 1, 0))

            speaker_id.insert(0, 0)
            speaker_id_idx = F.pad(speaker_id_idx, (1, 0))
            speaker_role.insert(0, None)
            speaker_role_idx = F.pad(speaker_role_idx, (1, 0))

            # TODO: This used to pad the sides. I don't think it's necessary to do this. confirm and remove this section
            # for side in segment_features_delta_sides:
            #     segment_features_sides[side] = F.pad(
            #         segment_features_sides[side], (0, 0, 1, 0)
            #     )
            #     segment_features_delta_sides[side] = F.pad(
            #         segment_features_delta_sides[side], (0, 0, 1, 0)
            #     )
            #     segment_features_sides_len[side][0] += 1

            transcript = [""] + transcript

        # Assemble prediction metadata and conversation side attention masks
        if self.role_type == PredictionType.AgentPartner:
            predict_next: Tensor = (
                speaker_role_idx == AgentPartnerPredictionRole.agent.value
            )[1:]
            history_mask_a = speaker_role_idx == AgentPartnerPredictionRole.agent.value
            history_mask_b = (
                speaker_role_idx == AgentPartnerPredictionRole.partner.value
            )
        elif self.role_type == PredictionType.Both:
            predict_next: Tensor = torch.tensor(
                [True for _ in range(len(speaker_role_idx) - 1)]
            )
            history_mask_a = speaker_role_idx == BothPredictionRole.a.value
            history_mask_b = speaker_role_idx == BothPredictionRole.b.value
        else:
            raise Exception("Oh no")

        if segment_embeddings is not None:
            segment_embeddings = segment_embeddings.unsqueeze(0)

        return ConversationData(
            conv_id=[conv_id],
            segment_features=segment_features.unsqueeze(0),
            segment_features_delta=segment_features_delta.unsqueeze(0),
            segment_features_sides=segment_features_sides,
            segment_features_sides_len=segment_features_sides_len,
            segment_features_delta_sides=segment_features_delta_sides,
            predict_next=predict_next.unsqueeze(0),
            history_mask_a=history_mask_a.unsqueeze(0),
            history_mask_b=history_mask_b.unsqueeze(0),
            word_embeddings=word_embeddings,
            embeddings_len=word_embeddings_len,
            segment_embeddings=segment_embeddings,
            num_segments=[segment_features.shape[0]],
            speaker_id=[speaker_id],
            speaker_id_idx=speaker_id_idx.unsqueeze(0),
            speaker_role=[speaker_role],
            speaker_role_idx=speaker_role_idx.unsqueeze(0),
            role_speaker_assignment=[role_speaker_assignment],
            role_speaker_assignment_idx=[role_speaker_assignment_idx],
            transcript=[transcript],
        )

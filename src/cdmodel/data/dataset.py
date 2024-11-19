import json
from os import path
from random import Random
from typing import Final

import pandas as pd
import torch
from torch import Tensor
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


def load_set_ids(dataset_dir: str, dataset_subset: str, set: str) -> list[int]:
    with open(path.join(dataset_dir, f"{set}-{dataset_subset}.csv")) as infile:
        return [int(x) for x in infile.readlines() if len(x) > 0]


class ConversationDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        segment_features: list[str],
        zero_pad: bool,
        subset: str,
        set: str,
        role_type: RoleType,
        role_assignment_strategy: RoleAssignmentStrategy,
        deterministic: bool = True,
    ):
        super().__init__()

        self.dataset_dir: Final[str] = dataset_dir
        self.conv_ids: Final[list[int]] = load_set_ids(
            dataset_dir=dataset_dir,
            dataset_subset=subset,
            set=set,
        )
        self.segment_features: Final[list[str]] = segment_features
        self.zero_pad: Final[bool] = zero_pad
        self.speaker_ids: Final[dict[int, int]] = pd.read_csv(
            path.join(dataset_dir, f"speaker-ids-{subset}.csv"),
            index_col="speaker_id",
        )["idx"].to_dict()
        self.role_type: Final[RoleType] = role_type
        self.role_assignment_strategy: Final[RoleAssignmentStrategy] = (
            role_assignment_strategy
        )

        self.deterministic: Final[bool] = deterministic
        self.random = Random()

        try:
            self.calldata = pd.concat(
                [
                    pd.read_csv(path.join(dataset_dir, "fe_03_p1_calldata.tbl")),
                    pd.read_csv(path.join(dataset_dir, "fe_03_p2_calldata.tbl")),
                ]
            ).reset_index(drop=True)

            # TODO: Make this part of the preprocessing
            self.pindata = pd.read_csv(
                path.join(dataset_dir, "fe_03_pindata.tbl"), index_col="PIN"
            )
        except:
            self.calldata = None
            self.pindata = None

    def __len__(self) -> int:
        return len(self.conv_ids)

    def __getitem__(self, i: int) -> ConversationData:
        conv_id: Final[int] = self.conv_ids[i]

        with open(path.join(self.dataset_dir, "segments", f"{conv_id}.json")) as infile:
            conv_data: Final[dict] = json.load(infile)

        segment_features: Tensor = torch.tensor(
            [conv_data[feature] for feature in self.segment_features]
        ).swapaxes(0, 1)
        segment_features_delta = segment_features.diff(
            dim=0, prepend=torch.zeros(1, segment_features.shape[1])
        )

        embeddings: Tensor = torch.load(
            path.join(self.dataset_dir, "embeddings", f"{conv_id}-embeddings.pt"),
            weights_only=True,
        )

        embeddings_turn_len: Tensor = torch.load(
            path.join(self.dataset_dir, "embeddings", f"{conv_id}-lengths.pt"),
            weights_only=True,
        )

        speaker_id: list[int] = conv_data["speaker_id"]
        speaker_id_idx: Tensor = torch.tensor(
            [self.speaker_ids[x] for x in conv_data["speaker_id"]],
            dtype=torch.long,
        )

        # Establish speaker roles
        if self.deterministic:
            self.random.seed(conv_id)
        speaker_role, role_speaker_assignment = assign_speaker_roles(
            speaker_ids=speaker_id,
            role_type=self.role_type,
            role_assignment_strategy=self.role_assignment_strategy,
            random=self.random,
        )
        speaker_role_idx = torch.tensor([x.value for x in speaker_role[0]])
        role_speaker_assignment_idx = {
            k: self.speaker_ids[v] for k, v in role_speaker_assignment.items()
        }
        try:
            role_speaker_assignment_gender = {}
            for assignment, id in role_speaker_assignment.items():
                gender_col = "ASX.DL"
                rows = self.calldata[self.calldata.APIN == id]
                if len(rows) == 0:
                    gender_col = "BSX.DL"
                    rows = self.calldata[self.calldata.BPIN == id]

                role_speaker_assignment_gender[assignment] = (
                    rows[gender_col].iloc[0].split(".")[0]
                )
            # role_speaker_assignment_gender = {
            #     k: (
            #         self.pindata.loc[v].S_SEX.lower()
            #         if not self.pindata.loc[v].S_SEX.S_SEX.isna()
            #         else "n"
            #     )
            #     for k, v in role_speaker_assignment.items()
            # }
        except:
            pass
            # for k, v in role_speaker_assignment.items():
            #     print(v)
            #     print(self.pindata.loc[v].S_SEX)

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
            speaker_role=speaker_role,
            speaker_role_idx=speaker_role_idx.unsqueeze(0),
            role_speaker_assignment=[role_speaker_assignment],
            role_speaker_assignment_idx=[role_speaker_assignment_idx],
            role_gender_assignment=[role_speaker_assignment_gender],
        )

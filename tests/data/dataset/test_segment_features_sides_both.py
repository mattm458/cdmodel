import torch
from torch.testing import assert_close

from cdmodel.common.role_assignment import (
    BothPredictionRole,
    PredictionType,
    RoleAssignmentStrategy,
)
from cdmodel.data.dataset import ConversationDataset
from tests.data.fixtures import *


def test_segment_features_sides_first_without_zero_padding(
    dataset_dir, feature_names, speaker_id_dict, segment_data_1
):
    """
    Ensure that outputted segment feature sides are correct for the following configuration:
        - Both prediction type
        - First role assignment strategy
        - No zero padding
    """
    expected_segment_features = torch.tensor(
        [segment_data_1[feature_name] for feature_name in feature_names]
    ).swapaxes(0, 1)
    expected_segment_features_delta = expected_segment_features.diff(
        dim=0, prepend=torch.zeros(1, expected_segment_features.shape[1])
    )

    speaker_ids = set(segment_data_1["speaker_id"])
    first_speaker_id = segment_data_1["speaker_id"][0]
    second_speaker_id = list(speaker_ids - {first_speaker_id})[0]

    # For the First role assignment strategy with AgentPartner role types,
    # the first speaker is designated the agent
    agent_turns = torch.tensor(segment_data_1["speaker_id"]) == first_speaker_id
    partner_turns = torch.tensor(segment_data_1["speaker_id"]) == second_speaker_id

    ds = ConversationDataset(
        dataset_dir=dataset_dir,
        feature_names=feature_names,
        zero_pad=False,
        role_type=PredictionType.Both,
        role_assignment_strategy=RoleAssignmentStrategy.first,
        conv_ids=[1],
        speaker_ids=speaker_id_dict,
        embeddings_type=None,
    )

    output = ds[0]

    assert len(ds) == 1

    assert output.segment_features_sides_len[BothPredictionRole.a] == [sum(agent_turns)]
    assert output.segment_features_sides_len[BothPredictionRole.b] == [
        sum(partner_turns)
    ]

    assert_close(
        expected_segment_features[agent_turns, :].unsqueeze(0),
        output.segment_features_sides[BothPredictionRole.a],
    )
    assert_close(
        expected_segment_features[partner_turns, :].unsqueeze(0),
        output.segment_features_sides[BothPredictionRole.b],
    )

    assert_close(
        expected_segment_features_delta[agent_turns, :].unsqueeze(0),
        output.segment_features_delta_sides[BothPredictionRole.a],
    )
    assert_close(
        expected_segment_features_delta[partner_turns, :].unsqueeze(0),
        output.segment_features_delta_sides[BothPredictionRole.b],
    )


def test_segment_features_sides_second_without_zero_padding(
    dataset_dir, feature_names, speaker_id_dict
):
    """
    Ensure that the dataloader raises an exception when the following configuration is given:
        - Both prediction type
        - Second role assignment strategy

    The Both prediction type can only be used with he First role assignment strategy
    """

    with pytest.raises(Exception):
        ConversationDataset(
            dataset_dir=dataset_dir,
            feature_names=feature_names,
            zero_pad=False,
            role_type=PredictionType.Both,
            role_assignment_strategy=RoleAssignmentStrategy.second,
            conv_ids=[1],
            speaker_ids=speaker_id_dict,
            embeddings_type=None,
        )

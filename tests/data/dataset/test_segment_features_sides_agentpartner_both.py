import torch
from torch.testing import assert_close

from cdmodel.common.role_assignment import (
    AgentPartnerPredictionRole,
    PredictionType,
    RoleAssignmentStrategy,
)
from cdmodel.data.dataset import ConversationDataset
from tests.data.fixtures import *


def test_segment_features_sides_both_without_zero_padding_0(
    dataset_dir, feature_names, speaker_id_dict, segment_data_1
):
    """
    Ensure that outputted segment feature sides are correct for the following configuration:
        - AgentPartner prediction type
        - Both role assignment strategy (1st instance is uses the First strategy)
        - No zero padding
    The Both role assignment strategy doubles the size of the dataset. Each instance appears twice: once with the First strategy, then again with the Second strategy.
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

    # The first instances uses the First role assignment strategy
    agent_turns = torch.tensor(segment_data_1["speaker_id"]) == first_speaker_id
    partner_turns = torch.tensor(segment_data_1["speaker_id"]) == second_speaker_id

    ds = ConversationDataset(
        dataset_dir=dataset_dir,
        feature_names=feature_names,
        zero_pad=False,
        role_type=PredictionType.AgentPartner,
        role_assignment_strategy=RoleAssignmentStrategy.both,
        conv_ids=[1],
        speaker_ids=speaker_id_dict,
        embeddings_type=None,
    )

    output = ds[0]

    assert len(ds) == 2

    assert output.segment_features_sides_len[AgentPartnerPredictionRole.agent] == [
        sum(agent_turns)
    ]
    assert output.segment_features_sides_len[AgentPartnerPredictionRole.partner] == [
        sum(partner_turns)
    ]

    assert_close(
        expected_segment_features[agent_turns, :].unsqueeze(0),
        output.segment_features_sides[AgentPartnerPredictionRole.agent],
    )
    assert_close(
        expected_segment_features[partner_turns, :].unsqueeze(0),
        output.segment_features_sides[AgentPartnerPredictionRole.partner],
    )

    assert_close(
        expected_segment_features_delta[agent_turns, :].unsqueeze(0),
        output.segment_features_delta_sides[AgentPartnerPredictionRole.agent],
    )
    assert_close(
        expected_segment_features_delta[partner_turns, :].unsqueeze(0),
        output.segment_features_delta_sides[AgentPartnerPredictionRole.partner],
    )


def test_segment_features_sides_both_with_zero_padding_0(
    dataset_dir, feature_names, speaker_id_dict, segment_data_1
):
    """
    Ensure that outputted segment feature sides are correct for the following configuration:
        - AgentPartner prediction type
        - Both role assignment strategy (1st instance is uses the First strategy)
        - Zero padding
    The Both role assignment strategy doubles the size of the dataset. Each instance appears twice: once with the First strategy, then again with the Second strategy.

    Zero padding should not affect this output.
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

    # The first instances uses the First role assignment strategy
    agent_turns = torch.tensor(segment_data_1["speaker_id"]) == first_speaker_id
    partner_turns = torch.tensor(segment_data_1["speaker_id"]) == second_speaker_id

    ds = ConversationDataset(
        dataset_dir=dataset_dir,
        feature_names=feature_names,
        zero_pad=True,
        role_type=PredictionType.AgentPartner,
        role_assignment_strategy=RoleAssignmentStrategy.both,
        conv_ids=[1],
        speaker_ids=speaker_id_dict,
        embeddings_type=None,
    )

    output = ds[0]

    assert len(ds) == 2

    assert output.segment_features_sides_len[AgentPartnerPredictionRole.agent] == [
        sum(agent_turns)
    ]
    assert output.segment_features_sides_len[AgentPartnerPredictionRole.partner] == [
        sum(partner_turns)
    ]

    assert_close(
        expected_segment_features[agent_turns, :].unsqueeze(0),
        output.segment_features_sides[AgentPartnerPredictionRole.agent],
    )
    assert_close(
        expected_segment_features[partner_turns, :].unsqueeze(0),
        output.segment_features_sides[AgentPartnerPredictionRole.partner],
    )

    assert_close(
        expected_segment_features_delta[agent_turns, :].unsqueeze(0),
        output.segment_features_delta_sides[AgentPartnerPredictionRole.agent],
    )
    assert_close(
        expected_segment_features_delta[partner_turns, :].unsqueeze(0),
        output.segment_features_delta_sides[AgentPartnerPredictionRole.partner],
    )


def test_segment_features_sides_both_without_zero_padding_1(
    dataset_dir, feature_names, speaker_id_dict, segment_data_1
):
    """
    Ensure that outputted segment feature sides are correct for the following configuration:
        - AgentPartner prediction type
        - Both role assignment strategy (1st instance is uses the First strategy)
        - No zero padding
    The Both role assignment strategy doubles the size of the dataset. Each instance appears twice: once with the First strategy, then again with the Second strategy.
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

    # The second instances uses the Second role assignment strategy
    agent_turns = torch.tensor(segment_data_1["speaker_id"]) == second_speaker_id
    partner_turns = torch.tensor(segment_data_1["speaker_id"]) == first_speaker_id

    ds = ConversationDataset(
        dataset_dir=dataset_dir,
        feature_names=feature_names,
        zero_pad=False,
        role_type=PredictionType.AgentPartner,
        role_assignment_strategy=RoleAssignmentStrategy.both,
        conv_ids=[1],
        speaker_ids=speaker_id_dict,
        embeddings_type=None,
    )

    output = ds[1]

    assert len(ds) == 2

    assert output.segment_features_sides_len[AgentPartnerPredictionRole.agent] == [
        sum(agent_turns)
    ]
    assert output.segment_features_sides_len[AgentPartnerPredictionRole.partner] == [
        sum(partner_turns)
    ]

    assert_close(
        expected_segment_features[agent_turns, :].unsqueeze(0),
        output.segment_features_sides[AgentPartnerPredictionRole.agent],
    )
    assert_close(
        expected_segment_features[partner_turns, :].unsqueeze(0),
        output.segment_features_sides[AgentPartnerPredictionRole.partner],
    )

    assert_close(
        expected_segment_features_delta[agent_turns, :].unsqueeze(0),
        output.segment_features_delta_sides[AgentPartnerPredictionRole.agent],
    )
    assert_close(
        expected_segment_features_delta[partner_turns, :].unsqueeze(0),
        output.segment_features_delta_sides[AgentPartnerPredictionRole.partner],
    )

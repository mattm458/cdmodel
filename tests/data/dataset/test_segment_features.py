import torch
from torch.testing import assert_close

from cdmodel.common.role_assignment import PredictionType, RoleAssignmentStrategy
from cdmodel.data.dataset import ConversationDataset
from tests.data.fixtures import *


def test_segment_features_without_zero_padding(
    dataset_dir, feature_names, speaker_id_dict, segment_data_1
):
    """
    Ensure that outputted segment features are as expected without zero padding.
    """
    expected_segment_features = torch.tensor(
        [segment_data_1[feature_name] for feature_name in feature_names]
    ).swapaxes(0, 1)
    expected_segment_features_delta = expected_segment_features.diff(
        dim=0, prepend=torch.zeros(1, expected_segment_features.shape[1])
    )

    ds = ConversationDataset(
        dataset_dir=dataset_dir,
        feature_names=feature_names,
        zero_pad=False,
        role_type=PredictionType.AgentPartner,
        role_assignment_strategy=RoleAssignmentStrategy.first,
        conv_ids=[1],
        speaker_ids=speaker_id_dict,
        embeddings_type=None,
    )

    output = ds[0]

    assert_close(expected_segment_features.unsqueeze(0), output.segment_features)
    assert_close(
        expected_segment_features_delta.unsqueeze(0), output.segment_features_delta
    )


def test_segment_features_with_zero_padding(
    dataset_dir, feature_names, speaker_id_dict, segment_data_1
):
    """
    Ensure that outputted segment features contain an initial zero segment.
    """
    expected_segment_features = torch.tensor(
        [[0] + segment_data_1[feature_name] for feature_name in feature_names]
    ).swapaxes(0, 1)
    expected_segment_features_delta = expected_segment_features.diff(
        dim=0, prepend=torch.zeros(1, expected_segment_features.shape[1])
    )

    ds = ConversationDataset(
        dataset_dir=dataset_dir,
        feature_names=feature_names,
        zero_pad=True,
        role_type=PredictionType.AgentPartner,
        role_assignment_strategy=RoleAssignmentStrategy.first,
        conv_ids=[1],
        speaker_ids=speaker_id_dict,
        embeddings_type=None,
    )

    output = ds[0]

    assert_close(expected_segment_features.unsqueeze(0), output.segment_features)
    assert_close(
        expected_segment_features_delta.unsqueeze(0), output.segment_features_delta
    )

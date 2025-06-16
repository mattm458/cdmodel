from cdmodel.common.role_assignment import PredictionType, RoleAssignmentStrategy
from cdmodel.data.dataset import ConversationDataset
from tests.data.fixtures import *


def test_metadata_without_zero_padding(dataset_dir, feature_names, speaker_id_dict):
    """
    Ensure that metadata reflects the correct number of segments in a conversation.
    """
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

    assert len(ds) == 1
    assert output.conv_id == [1]
    assert output.num_segments == [9]


def test_metadata_with_zero_padding(dataset_dir, feature_names, speaker_id_dict):
    """
    Ensure that zero padding is correctly accounted for in the number of segments reported in conversation metadata.
    """
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

    assert len(ds) == 1
    assert output.conv_id == [1]
    assert output.num_segments == [10]

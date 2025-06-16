from cdmodel.common.role_assignment import PredictionType, RoleAssignmentStrategy
from cdmodel.data.dataset import ConversationDataset
from tests.data.fixtures import *


def test_transcript_without_zero_padding(
    dataset_dir, feature_names, speaker_id_dict, segment_data_1
):
    """
    Ensure that the transcript is correctly outputted without zero padding.
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

    assert output.transcript == [segment_data_1["transcript"]]


def test_transcript_with_zero_padding(
    dataset_dir, feature_names, speaker_id_dict, segment_data_1
):
    """
    Ensure that the transcript is correctly zero-padded with an empty string.
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

    assert output.transcript == [[""] + segment_data_1["transcript"]]

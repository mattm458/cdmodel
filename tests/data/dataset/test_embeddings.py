from torch.nn import functional as F
from torch.testing import assert_close

from cdmodel.common.role_assignment import PredictionType, RoleAssignmentStrategy
from cdmodel.data.dataset import ConversationDataset
from tests.data.fixtures import *


def test_segment_embeddings_without_zero_pad(
    dataset_dir,
    feature_names,
    speaker_id_dict,
    segment_embeddings_1,
):
    """
    Ensure that segment embeddings are correctly output without zero padding
    """
    ds = ConversationDataset(
        dataset_dir=dataset_dir,
        feature_names=feature_names,
        zero_pad=False,
        role_type=PredictionType.AgentPartner,
        role_assignment_strategy=RoleAssignmentStrategy.first,
        conv_ids=[1],
        speaker_ids=speaker_id_dict,
        embeddings_type="roberta",
    )

    output = ds[0]

    assert_close(output.segment_embeddings, segment_embeddings_1.unsqueeze(0))


def test_segment_embeddings_with_zero_pad(
    dataset_dir,
    feature_names,
    speaker_id_dict,
    segment_embeddings_1,
):
    """
    Ensure that segment embeddings are correctly output with zero padding
    """
    ds = ConversationDataset(
        dataset_dir=dataset_dir,
        feature_names=feature_names,
        zero_pad=True,
        role_type=PredictionType.AgentPartner,
        role_assignment_strategy=RoleAssignmentStrategy.first,
        conv_ids=[1],
        speaker_ids=speaker_id_dict,
        embeddings_type="roberta",
    )

    output = ds[0]

    assert_close(
        output.segment_embeddings,
        F.pad(segment_embeddings_1, (0, 0, 1, 0)).unsqueeze(0),
    )

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.testing import assert_close

from cdmodel.common.role_assignment import PredictionType, RoleAssignmentStrategy
from cdmodel.data.collate_fn import collate_fn
from cdmodel.data.dataset import ConversationDataset
from tests.data.fixtures import *


def test_collate_fn_without_embeddings(dataset_dir, feature_names, speaker_id_dict):
    """
    Ensure that the collate function successfully outputs None if the dataset does not output any word embeddings.
    """
    ds = ConversationDataset(
        dataset_dir=dataset_dir,
        feature_names=feature_names,
        zero_pad=False,
        role_type=PredictionType.AgentPartner,
        role_assignment_strategy=RoleAssignmentStrategy.first,
        conv_ids=[1, 2],
        speaker_ids=speaker_id_dict,
        embeddings_type=None,
    )

    output = collate_fn([ds[0], ds[1]])

    assert output.word_embeddings is None
    assert output.embeddings_len is None
    assert output.segment_embeddings is None


def test_collate_fn_with_word_embeddings(
    dataset_dir,
    feature_names,
    speaker_id_dict,
    word_embeddings_1,
    word_embeddings_2,
    word_embeddings_len_1,
    word_embeddings_len_2,
):
    """
    Ensure that the collate function successfully outputs word-level embeddings and their lengths
    """
    ds = ConversationDataset(
        dataset_dir=dataset_dir,
        feature_names=feature_names,
        zero_pad=False,
        role_type=PredictionType.AgentPartner,
        role_assignment_strategy=RoleAssignmentStrategy.first,
        conv_ids=[1, 2],
        speaker_ids=speaker_id_dict,
        embeddings_type="glove",
    )

    output = collate_fn([ds[0], ds[1]])

    # Word embeddings 1 is (9,28,50) and 2 is (14,31,50).
    # To recreate what the collate function does, we have to pad 1 along dimension 2
    # so it is the same size as 2: (9,28,50) -> (9,31,50).
    # Then, we concat both along the 0th dimension to make a (23,31,50) Tensor.
    expected_word_embeddings = torch.concat(
        [F.pad(word_embeddings_1, (0, 0, 0, 3)), word_embeddings_2], dim=0
    )

    assert_close(output.word_embeddings, expected_word_embeddings)
    assert_close(
        output.embeddings_len,
        torch.cat([word_embeddings_len_1, word_embeddings_len_2]),
    )
    assert output.segment_embeddings is None


def test_collate_fn_with_segment_embeddings(
    dataset_dir,
    feature_names,
    speaker_id_dict,
    segment_embeddings_1,
    segment_embeddings_2,
):
    """
    Ensure that the collate function successfully outputs segment-level embeddings and their lengths
    """
    ds = ConversationDataset(
        dataset_dir=dataset_dir,
        feature_names=feature_names,
        zero_pad=False,
        role_type=PredictionType.AgentPartner,
        role_assignment_strategy=RoleAssignmentStrategy.first,
        conv_ids=[1, 2],
        speaker_ids=speaker_id_dict,
        embeddings_type="roberta",
    )

    output = collate_fn([ds[0], ds[1]])

    assert_close(
        output.segment_embeddings,
        pad_sequence(
            [segment_embeddings_1, segment_embeddings_2],
            batch_first=True,
        ),
    )
    assert_close(
        output.embeddings_len,
        torch.tensor([segment_embeddings_1.shape[0], segment_embeddings_2.shape[0]]),
    )

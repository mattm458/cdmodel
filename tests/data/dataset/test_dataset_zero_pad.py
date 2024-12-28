import pytest
import torch
from torch.testing import assert_close

from cdmodel.common.role_assignment import (
    DialogueSystemRole,
    RoleAssignmentStrategy,
    RoleType,
)
from cdmodel.data.dataset import ConversationDataset


# Fixtures
# ========================================================
@pytest.fixture
def speaker_id_dict():
    return {1: 1000, 2: 1001}


@pytest.fixture
def segment_data_tensor():
    return torch.tensor(
        [
            [1, 2, 4, 7, 11, 16, 15, 13, 10],
            [10, 20, 40, 70, 110, 160, 150, 130, 100],
            [100, 200, 400, 700, 1100, 1600, 1500, 1300, 1000],
        ]
    ).T


@pytest.fixture
def segment_data_delta_tensor(segment_data_tensor):
    return segment_data_tensor.diff(
        dim=0, prepend=torch.zeros(1, segment_data_tensor.shape[1])
    )


@pytest.fixture
def speaker_id_data():
    return [1, 2, 1, 1, 2, 2, 1, 2, 1]


@pytest.fixture
def speaker_role_ds_second(speaker_id_data):
    partner_id = speaker_id_data[0]
    return [
        DialogueSystemRole.partner if x == partner_id else DialogueSystemRole.agent
        for x in speaker_id_data
    ]


@pytest.fixture
def speaker_role_ds_first(speaker_id_data):
    agent_id = speaker_id_data[0]
    return [
        DialogueSystemRole.agent if x == agent_id else DialogueSystemRole.partner
        for x in speaker_id_data
    ]


@pytest.fixture
def speaker_role_ds_second_idx_tensor(speaker_role_ds_second):
    return torch.tensor([x.value for x in speaker_role_ds_second])


@pytest.fixture
def speaker_role_ds_first_idx_tensor(speaker_role_ds_first):
    return torch.tensor([x.value for x in speaker_role_ds_first])


@pytest.fixture
def speaker_id_idx_data_tensor(speaker_id_data, speaker_id_dict):
    return torch.tensor([speaker_id_dict[x] for x in speaker_id_data])


@pytest.fixture
def segment_data(segment_data_tensor, speaker_id_data):
    return {
        "feature_a": segment_data_tensor[:, 0].tolist(),
        "feature_b": segment_data_tensor[:, 1].tolist(),
        "feature_c": segment_data_tensor[:, 2].tolist(),
        "speaker_id": speaker_id_data,
    }


@pytest.fixture
def embedding_data():
    return torch.zeros(9, 10, 32)


@pytest.fixture
def embedding_len_data():
    return torch.tensor([3, 5, 2, 6, 10, 6, 5, 4, 9])


@pytest.fixture
def conversation_dataset_ds_second_zero_pad(speaker_id_dict):
    return ConversationDataset(
        dataset_dir="test",
        segment_features=["feature_a", "feature_b", "feature_c"],
        zero_pad=True,
        role_type=RoleType.DialogueSystem,
        role_assignment_strategy=RoleAssignmentStrategy.second,
        conv_ids=[1, 2, 3],
        speaker_ids=speaker_id_dict,
        deterministic=False,
    )


@pytest.fixture
def conversation_dataset_ds_first_zero_pad(speaker_id_dict):
    return ConversationDataset(
        dataset_dir="test",
        segment_features=["feature_a", "feature_b", "feature_c"],
        zero_pad=True,
        role_type=RoleType.DialogueSystem,
        role_assignment_strategy=RoleAssignmentStrategy.first,
        conv_ids=[1, 2, 3],
        speaker_ids=speaker_id_dict,
        deterministic=False,
    )


# Tests
# ========================================================
def test__dataset_ds_second_zero_pad(
    mocker,
    conversation_dataset_ds_second_zero_pad,
    segment_data,
    segment_data_tensor,
    segment_data_delta_tensor,
    embedding_data,
    embedding_len_data,
    speaker_id_data,
    speaker_id_idx_data_tensor,
    speaker_role_ds_second,
    speaker_role_ds_second_idx_tensor,
):
    load_segment_data_mock = mocker.patch("cdmodel.data.dataset._load_segment_data")
    load_segment_data_mock.return_value = segment_data

    load_embeddings_mock = mocker.patch("cdmodel.data.dataset._load_embeddings")
    load_embeddings_mock.return_value = (
        embedding_data,
        embedding_len_data,
    )

    num_turns = len(segment_data["feature_a"])

    output = conversation_dataset_ds_second_zero_pad[0]

    # Ensure that unpaddable output was not padded
    assert output.conv_id == [1]

    # Ensure turn-level features were padded
    assert output.num_segments == [num_turns + 1]

    # Ensure the first turn in the segment features is 0
    assert_close(
        output.segment_features[:, 0], torch.zeros_like(output.segment_features[:, 0])
    )
    assert_close(
        output.segment_features_delta[:, 0],
        torch.zeros_like(output.segment_features_delta[:, 0]),
    )

    # Ensure the remaining segment features are identical to the original
    assert_close(output.segment_features[0, 1:], segment_data_tensor)
    assert_close(output.segment_features_delta[0, 1:], segment_data_delta_tensor)

    # Ensure the embeddings were padded
    assert_close(output.embeddings[0], torch.zeros_like(embedding_data[0]))
    assert_close(output.embeddings[1:], embedding_data)

    # Ensure the embedding lengths were padded
    assert output.embeddings_segment_len[0] == 1
    assert_close(output.embeddings_segment_len[1:], embedding_len_data)

    # Ensure the speaker IDs and their indices were padded
    assert output.speaker_id[0][0] == 0
    assert output.speaker_id_idx[0, 0] == 0

    assert output.speaker_id[0][1:] == speaker_id_data
    assert_close(output.speaker_id_idx[0, 1:], speaker_id_idx_data_tensor)

    assert output.speaker_role[0][0] == None
    assert output.speaker_role[0][1:] == speaker_role_ds_second

    assert output.speaker_role_idx[0, 0] == 0
    assert_close(output.speaker_role_idx[0, 1:], speaker_role_ds_second_idx_tensor)


def test__dataset_ds_first_zero_pad(
    mocker,
    conversation_dataset_ds_first_zero_pad,
    segment_data,
    segment_data_tensor,
    segment_data_delta_tensor,
    embedding_data,
    embedding_len_data,
    speaker_id_data,
    speaker_id_idx_data_tensor,
    speaker_role_ds_first,
    speaker_role_ds_first_idx_tensor,
):
    load_segment_data_mock = mocker.patch("cdmodel.data.dataset._load_segment_data")
    load_segment_data_mock.return_value = segment_data

    load_embeddings_mock = mocker.patch("cdmodel.data.dataset._load_embeddings")
    load_embeddings_mock.return_value = (
        embedding_data,
        embedding_len_data,
    )

    num_turns = len(segment_data["feature_a"])

    output = conversation_dataset_ds_first_zero_pad[0]

    # Ensure that unpaddable output was not padded
    assert output.conv_id == [1]

    # Ensure turn-level features were padded
    assert output.num_segments == [num_turns + 1]

    # Ensure the first turn in the segment features is 0
    assert_close(
        output.segment_features[:, 0], torch.zeros_like(output.segment_features[:, 0])
    )
    assert_close(
        output.segment_features_delta[:, 0],
        torch.zeros_like(output.segment_features_delta[:, 0]),
    )

    # Ensure the remaining segment features are identical to the original
    assert_close(output.segment_features[0, 1:], segment_data_tensor)
    assert_close(output.segment_features_delta[0, 1:], segment_data_delta_tensor)

    # Ensure the embeddings were padded
    assert_close(output.embeddings[0], torch.zeros_like(embedding_data[0]))
    assert_close(output.embeddings[1:], embedding_data)

    # Ensure the embedding lengths were padded
    assert output.embeddings_segment_len[0] == 1
    assert_close(output.embeddings_segment_len[1:], embedding_len_data)

    # Ensure the speaker IDs and their indices were padded
    assert output.speaker_id[0][0] == 0
    assert output.speaker_id_idx[0, 0] == 0

    assert output.speaker_id[0][1:] == speaker_id_data
    assert_close(output.speaker_id_idx[0, 1:], speaker_id_idx_data_tensor)

    assert output.speaker_role[0][0] == None
    assert output.speaker_role[0][1:] == speaker_role_ds_first

    assert output.speaker_role_idx[0, 0] == 0
    assert_close(output.speaker_role_idx[0, 1:], speaker_role_ds_first_idx_tensor)

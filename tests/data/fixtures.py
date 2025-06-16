import json
from os import path

import pytest

from cdmodel.data.dataset import load_embeddings, load_embeddings_len


@pytest.fixture
def dataset_dir(pytestconfig):
    return path.join(pytestconfig.rootpath, "tests/fixtures/dataset")


@pytest.fixture
def feature_names():
    return ("pitch_mean", "pitch_range", "intensity_mean")


@pytest.fixture
def speaker_id_dict():
    return {1001: 0, 1002: 1, 1003: 2}


@pytest.fixture
def segment_data_1(pytestconfig):
    with open(
        path.join(pytestconfig.rootpath, "tests/fixtures/dataset/segments/1.json")
    ) as infile:
        return json.load(infile)


@pytest.fixture
def segment_data_2(pytestconfig):
    with open(
        path.join(pytestconfig.rootpath, "tests/fixtures/dataset/segments/2.json")
    ) as infile:
        return json.load(infile)


@pytest.fixture
def word_embeddings_1(pytestconfig):
    return load_embeddings(
        path.join(pytestconfig.rootpath, "tests/fixtures/dataset"), 1, "glove"
    )


@pytest.fixture
def word_embeddings_2(pytestconfig):
    return load_embeddings(
        path.join(pytestconfig.rootpath, "tests/fixtures/dataset"), 2, "glove"
    )


@pytest.fixture
def segment_embeddings_1(pytestconfig):
    return load_embeddings(
        path.join(pytestconfig.rootpath, "tests/fixtures/dataset"), 1, "roberta"
    )


@pytest.fixture
def segment_embeddings_2(pytestconfig):
    return load_embeddings(
        path.join(pytestconfig.rootpath, "tests/fixtures/dataset"), 2, "roberta"
    )


@pytest.fixture
def word_embeddings_len_1(pytestconfig):
    return load_embeddings_len(
        path.join(pytestconfig.rootpath, "tests/fixtures/dataset"), 1, "glove"
    )


@pytest.fixture
def word_embeddings_len_2(pytestconfig):
    return load_embeddings_len(
        path.join(pytestconfig.rootpath, "tests/fixtures/dataset"), 2, "glove"
    )

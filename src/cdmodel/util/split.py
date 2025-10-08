import networkx as nx
from collections import defaultdict
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split


def __get_disjoint_speaker_groups(df):
    # Construct a graph consisting of conversation nodes.
    # The conversation nodes are connected if they have at least one speaker in common
    G = nx.Graph()
    spk_conv_dict_set = defaultdict(set)
    for row in df[["conv_id", "spk_id"]].drop_duplicates().itertuples():
        G.add_node(row.conv_id)
        spk_conv_dict_set[row.spk_id].add(row.conv_id)

    for conv_ids in spk_conv_dict_set.values():
        G.add_edges_from(list(itertools.combinations(list(conv_ids), r=2)))

    # Find groups of connected conversations. Return a new DataFrame recording
    # each conversation, a unique ID identifying the group it belongs to,
    # and the size of the group
    return pd.DataFrame(
        list(
            itertools.chain(
                *[
                    [(y, len(x), i + 1) for y in x]
                    for i, x in enumerate(nx.connected_components(G))
                ]
            )
        ),
        columns=("conv_id", "connected_group_size", "group"),
    )


def __build_disjoint_group_set(choices_df, size: float, total_conv_ids: int):
    groups: list[int] = []
    conv_ids: list[int] = []

    group_choices = (
        choices_df.sort_values("connected_group_size", ascending=False)
        .group.unique()
        .tolist()
    )

    group_members = defaultdict(list)
    for row in choices_df[choices_df.group.isin(group_choices)].itertuples():
        group_members[row.group].append(row.conv_id)

    while len(conv_ids) / total_conv_ids <= size and len(group_choices) > 0:
        choice = group_choices.pop()
        groups.append(choice)
        conv_ids.extend(group_members[choice])

    return groups, conv_ids


def split_disjoint(
    df: pd.DataFrame, train_size=0.9, val_size=0.05, test_size=0.05, group_cutoff=3
) -> tuple[list[int], list[int], list[int]]:
    connected_df = __get_disjoint_speaker_groups(df)

    testing_groups = []
    testing_conv_ids = []
    x = 1.0

    low_connected_df = connected_df[connected_df.connected_group_size <= group_cutoff]
    high_connected_df = connected_df[connected_df.connected_group_size > group_cutoff]

    eligible_groups_df = low_connected_df

    while len(testing_conv_ids) / len(connected_df) < val_size + test_size:
        high_sample_conv_id = high_connected_df.sample(
            frac=1.0 - x, random_state=42, replace=False
        ).conv_id.tolist()

        eligible_groups_df = pd.concat(
            [
                low_connected_df,
                __get_disjoint_speaker_groups(df[df.conv_id.isin(high_sample_conv_id)]),
            ]
        )

        testing_groups, testing_conv_ids = __build_disjoint_group_set(
            eligible_groups_df,
            size=val_size + test_size,
            total_conv_ids=len(connected_df.conv_id.unique()),
        )

        x /= 2

    val_groups, test_groups = train_test_split(
        testing_groups, test_size=test_size / (test_size + val_size), random_state=42
    )

    val_conv_ids = eligible_groups_df[
        eligible_groups_df.group.isin(val_groups)
    ].conv_id.tolist()
    test_conv_ids = eligible_groups_df[
        eligible_groups_df.group.isin(test_groups)
    ].conv_id.tolist()
    train_conv_ids = connected_df[
        ~connected_df.conv_id.isin(val_conv_ids + test_conv_ids)
    ].conv_id.tolist()

    return train_conv_ids, val_conv_ids, test_conv_ids


def split_by_id(
    ids: set[int], train_size=0.9, val_size=0.05, test_size=0.05
) -> tuple[list[int], list[int], list[int]]:
    ids_lst = list(ids)
    ids_lst.sort()

    ids_train, ids_test = train_test_split(
        ids_lst,
        random_state=42,
        train_size=train_size,
    )

    ids_test, ids_val = train_test_split(
        ids_test, random_state=42, test_size=test_size / (test_size + val_size)
    )

    return ids_train, ids_val, ids_test

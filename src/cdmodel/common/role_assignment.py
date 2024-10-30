from enum import Enum
from random import Random

RoleAssignmentStrategy = Enum("RoleAssignmentStrategy", ["first", "second", "random"])
# TODO lowercase these enum labels
RoleType = Enum("RoleType", ["DialogueSystem", "Analysis"])


class Role(Enum):
    pass


class DialogueSystemRole(Role):
    partner = 1
    agent = 2


class AnalysisRole(Role):
    a = 1
    b = 2


def __get_speaker_role_assignment_ds(
    first_speaker_id: int,
    second_speaker_id: int,
    role_assignment_strategy: RoleAssignmentStrategy,
    random: Random,
) -> dict[int, DialogueSystemRole]:
    match role_assignment_strategy:
        case RoleAssignmentStrategy.first:
            first_role = DialogueSystemRole.agent
            second_role = DialogueSystemRole.partner
        case RoleAssignmentStrategy.second:
            first_role = DialogueSystemRole.partner
            second_role = DialogueSystemRole.agent
        case _:  # Random strategy
            roles = [DialogueSystemRole.agent, DialogueSystemRole.partner]
            random.shuffle(roles)
            first_role, second_role = roles

    return {
        first_speaker_id: first_role,
        second_speaker_id: second_role,
    }


def __get_speaker_role_assignment_a(
    first_speaker_id: int,
    second_speaker_id: int,
    role_assignment_strategy: RoleAssignmentStrategy,
    random: Random,
) -> dict[int, AnalysisRole]:
    match role_assignment_strategy:
        case RoleAssignmentStrategy.first:
            first_role = AnalysisRole.a
            second_role = AnalysisRole.b
        case RoleAssignmentStrategy.second:
            first_role = AnalysisRole.b
            second_role = AnalysisRole.a
        case _:  # Random strategy
            roles = [AnalysisRole.a, AnalysisRole.b]
            random.shuffle(roles)
            first_role, second_role = roles

    return {
        first_speaker_id: first_role,
        second_speaker_id: second_role,
    }


def assign_speaker_roles(
    speaker_ids: list[int],
    role_type: RoleType,
    role_assignment_strategy: RoleAssignmentStrategy,
    random: Random,
) -> list[list[DialogueSystemRole]] | list[list[AnalysisRole]]:
    first_speaker_id = speaker_ids[0]
    second_speaker_id = list(set(speaker_ids) - set([speaker_ids[0]]))[0]

    match role_type:
        case RoleType.DialogueSystem:
            assignments_ds = __get_speaker_role_assignment_ds(
                first_speaker_id, second_speaker_id, role_assignment_strategy, random
            )
            return [[assignments_ds[x] for x in speaker_ids]]

        case RoleType.Analysis:
            assignments_a = __get_speaker_role_assignment_a(
                first_speaker_id, second_speaker_id, role_assignment_strategy, random
            )
            return [[assignments_a[x] for x in speaker_ids]]

        case _:
            raise NotImplementedError(f"Role type {role_type} is not implemented")

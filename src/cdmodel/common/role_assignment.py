from enum import Enum
from random import Random
from typing import Optional

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


def _get_speaker_role_assignment_dataset(
    first_speaker_id: int,
    second_speaker_id: int,
    role_assignment_strategy: RoleAssignmentStrategy,
    random: Optional[Random] = None,
) -> dict[int, DialogueSystemRole]:
    match role_assignment_strategy:
        case RoleAssignmentStrategy.first:
            first_role = DialogueSystemRole.agent
            second_role = DialogueSystemRole.partner
        case RoleAssignmentStrategy.second:
            first_role = DialogueSystemRole.partner
            second_role = DialogueSystemRole.agent
        case RoleAssignmentStrategy.random:
            if random is None:
                raise ValueError(
                    "If using the random role assignment strategy, a Random instance must be provided"
                )
            roles = [DialogueSystemRole.agent, DialogueSystemRole.partner]
            random.shuffle(roles)
            first_role, second_role = roles
        case _:
            raise NotImplementedError(
                f"Unimplemented role assignment strategy '{role_assignment_strategy}'"
            )

    return {
        first_speaker_id: first_role,
        second_speaker_id: second_role,
    }


def _get_speaker_role_assignment_analysis(
    first_speaker_id: int,
    second_speaker_id: int,
    role_assignment_strategy: RoleAssignmentStrategy,
    random: Optional[Random] = None,
) -> dict[int, AnalysisRole]:
    match role_assignment_strategy:
        case RoleAssignmentStrategy.first:
            first_role = AnalysisRole.a
            second_role = AnalysisRole.b
        case RoleAssignmentStrategy.second:
            first_role = AnalysisRole.b
            second_role = AnalysisRole.a
        case _:  # Random strategy
            if random is None:
                raise ValueError(
                    "If using the random role assignment strategy, a Random instance must be provided"
                )

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
    random: Optional[Random] = None,
) -> tuple[list[Role | None], dict[Role, int]]:
    first_speaker_id = speaker_ids[0]
    second_speaker_id = list(set(speaker_ids) - set([first_speaker_id]))[0]

    match role_type:
        case RoleType.DialogueSystem:
            assignments_dataset = _get_speaker_role_assignment_dataset(
                first_speaker_id=first_speaker_id,
                second_speaker_id=second_speaker_id,
                role_assignment_strategy=role_assignment_strategy,
                random=random,
            )
            return [assignments_dataset[x] for x in speaker_ids], {
                v: k for k, v in assignments_dataset.items()
            }

        case RoleType.Analysis:
            assignments_analysis = _get_speaker_role_assignment_analysis(
                first_speaker_id=first_speaker_id,
                second_speaker_id=second_speaker_id,
                role_assignment_strategy=role_assignment_strategy,
                random=random,
            )
            return [assignments_analysis[x] for x in speaker_ids], {
                v: k for k, v in assignments_analysis.items()
            }

        case _:
            raise NotImplementedError(f"Role type {role_type} is not implemented")

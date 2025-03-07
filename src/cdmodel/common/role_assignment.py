from enum import Enum

RoleAssignmentStrategy = Enum("RoleAssignmentStrategy", ["first", "second", "both"])

# TODO lowercase these enum labels
PredictionType = Enum("PredictionType", ["AgentPartner", "Both"])


class Role(Enum):
    pass


class AgentPartnerPredictionRole(Role):
    partner = 1
    agent = 2


class BothPredictionRole(Role):
    a = 1
    b = 2


def _get_speaker_role_assignment_dataset(
    first_speaker_id: int,
    second_speaker_id: int,
    role_assignment_strategy: RoleAssignmentStrategy,
) -> dict[int, AgentPartnerPredictionRole]:
    match role_assignment_strategy:
        case RoleAssignmentStrategy.first:
            first_role = AgentPartnerPredictionRole.agent
            second_role = AgentPartnerPredictionRole.partner
        case RoleAssignmentStrategy.second:
            first_role = AgentPartnerPredictionRole.partner
            second_role = AgentPartnerPredictionRole.agent
        case RoleAssignmentStrategy.both:
            raise NotImplementedError(
                "With 'both'' role assignment, the caller is responsible for calling _get_role_assignment_dataset() with 'first' and 'second' separately"
            )
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
) -> dict[int, BothPredictionRole]:
    match role_assignment_strategy:
        case RoleAssignmentStrategy.first:
            first_role = BothPredictionRole.a
            second_role = BothPredictionRole.b
        case RoleAssignmentStrategy.second:
            first_role = BothPredictionRole.b
            second_role = BothPredictionRole.a
        case RoleAssignmentStrategy.both:
            raise NotImplementedError(
                "With 'both'' role assignment, the caller is responsible for calling _get_speaker_role_assignment_analysis() with 'first' and 'second' separately"
            )
        case _:
            raise NotImplementedError(
                f"Unimplemented role assignment strategy '{role_assignment_strategy}'"
            )

    return {
        first_speaker_id: first_role,
        second_speaker_id: second_role,
    }


def assign_speaker_roles(
    speaker_ids: list[int],
    prediction_type: PredictionType,
    role_assignment_strategy: RoleAssignmentStrategy,
) -> tuple[list[Role | None], dict[Role, int]]:
    first_speaker_id = speaker_ids[0]
    second_speaker_id = list(set(speaker_ids) - set([first_speaker_id]))[0]

    match prediction_type:
        case PredictionType.AgentPartner:
            assignments_dataset = _get_speaker_role_assignment_dataset(
                first_speaker_id=first_speaker_id,
                second_speaker_id=second_speaker_id,
                role_assignment_strategy=role_assignment_strategy,
            )
            return [assignments_dataset[x] for x in speaker_ids], {
                v: k for k, v in assignments_dataset.items()
            }

        case PredictionType.Both:
            if role_assignment_strategy != RoleAssignmentStrategy.first:
                raise Exception(
                    "The 'Both' prediction type must be used with the 'first' role assignment strategy!"
                )

            assignments_analysis = _get_speaker_role_assignment_analysis(
                first_speaker_id=first_speaker_id,
                second_speaker_id=second_speaker_id,
                role_assignment_strategy=RoleAssignmentStrategy.first,
            )
            return [assignments_analysis[x] for x in speaker_ids], {
                v: k for k, v in assignments_analysis.items()
            }

        case _:
            raise NotImplementedError(
                f"Prediction type {prediction_type} is not implemented"
            )

from cdmodel.common.role_assignment import (
    AgentPartnerPredictionRole,
    RoleAssignmentStrategy,
    PredictionType,
    assign_speaker_roles,
)


def test_assign_speaker_roles_dialogue_first():
    assert assign_speaker_roles(
        speaker_ids=[1, 2, 2, 1, 2],
        prediction_type=PredictionType.AgentPartner,
        role_assignment_strategy=RoleAssignmentStrategy.first,
    ) == (
        [
            AgentPartnerPredictionRole.agent,
            AgentPartnerPredictionRole.partner,
            AgentPartnerPredictionRole.partner,
            AgentPartnerPredictionRole.agent,
            AgentPartnerPredictionRole.partner,
        ],
        {AgentPartnerPredictionRole.agent: 1, AgentPartnerPredictionRole.partner: 2},
    )


def test_assign_speaker_roles_dialogue_second():
    assert assign_speaker_roles(
        speaker_ids=[1, 2, 2, 1, 2],
        prediction_type=PredictionType.AgentPartner,
        role_assignment_strategy=RoleAssignmentStrategy.second,
    ) == (
        [
            AgentPartnerPredictionRole.partner,
            AgentPartnerPredictionRole.agent,
            AgentPartnerPredictionRole.agent,
            AgentPartnerPredictionRole.partner,
            AgentPartnerPredictionRole.agent,
        ],
        {AgentPartnerPredictionRole.agent: 2, AgentPartnerPredictionRole.partner: 1},
    )

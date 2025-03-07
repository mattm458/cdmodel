import pytest

from cdmodel.common.role_assignment import (
    AgentPartnerPredictionRole,
    RoleAssignmentStrategy,
    _get_speaker_role_assignment_dataset,
)


def test__get_speaker_role_assignment_dataset__first():
    assert _get_speaker_role_assignment_dataset(
        first_speaker_id=1,
        second_speaker_id=2,
        role_assignment_strategy=RoleAssignmentStrategy.first,
    ) == {1: AgentPartnerPredictionRole.agent, 2: AgentPartnerPredictionRole.partner}


def test__get_speaker_role_assignment_dataset__second():
    assert _get_speaker_role_assignment_dataset(
        first_speaker_id=1,
        second_speaker_id=2,
        role_assignment_strategy=RoleAssignmentStrategy.second,
    ) == {1: AgentPartnerPredictionRole.partner, 2: AgentPartnerPredictionRole.agent}


def test__get_speaker_role_assignment_dataset__both_exception(mocker):
    with pytest.raises(NotImplementedError) as e:
        _get_speaker_role_assignment_dataset(
            first_speaker_id=1,
            second_speaker_id=2,
            role_assignment_strategy=RoleAssignmentStrategy.both,
        )

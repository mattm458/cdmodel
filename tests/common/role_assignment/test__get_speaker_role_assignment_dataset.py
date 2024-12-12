from cdmodel.common.role_assignment import (
    DialogueSystemRole,
    RoleAssignmentStrategy,
    _get_speaker_role_assignment_dataset,
)


def test__get_speaker_role_assignment_dataset__first():
    assert _get_speaker_role_assignment_dataset(
        first_speaker_id=1,
        second_speaker_id=2,
        role_assignment_strategy=RoleAssignmentStrategy.first,
    ) == {1: DialogueSystemRole.agent, 2: DialogueSystemRole.partner}


def test__get_speaker_role_assignment_dataset__second():
    assert _get_speaker_role_assignment_dataset(
        first_speaker_id=1,
        second_speaker_id=2,
        role_assignment_strategy=RoleAssignmentStrategy.second,
    ) == {1: DialogueSystemRole.partner, 2: DialogueSystemRole.agent}


def test__get_speaker_role_assignment_dataset__random(mocker):
    mock = mocker.MagicMock()

    assert _get_speaker_role_assignment_dataset(
        first_speaker_id=1,
        second_speaker_id=2,
        role_assignment_strategy=RoleAssignmentStrategy.random,
        random=mock,
    ) == {1: DialogueSystemRole.agent, 2: DialogueSystemRole.partner}

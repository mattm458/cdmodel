from cdmodel.common.role_assignment import (
    DialogueSystemRole,
    RoleAssignmentStrategy,
    RoleType,
    assign_speaker_roles,
)


def test_assign_speaker_roles_dialogue_first():
    assert assign_speaker_roles(
        speaker_ids=[1, 2, 2, 1, 2],
        role_type=RoleType.DialogueSystem,
        role_assignment_strategy=RoleAssignmentStrategy.first,
    ) == (
        [
            DialogueSystemRole.agent,
            DialogueSystemRole.partner,
            DialogueSystemRole.partner,
            DialogueSystemRole.agent,
            DialogueSystemRole.partner,
        ],
        {DialogueSystemRole.agent: 1, DialogueSystemRole.partner: 2},
    )


def test_assign_speaker_roles_dialogue_second():
    assert assign_speaker_roles(
        speaker_ids=[1, 2, 2, 1, 2],
        role_type=RoleType.DialogueSystem,
        role_assignment_strategy=RoleAssignmentStrategy.second,
    ) == (
        [
            DialogueSystemRole.partner,
            DialogueSystemRole.agent,
            DialogueSystemRole.agent,
            DialogueSystemRole.partner,
            DialogueSystemRole.agent,
        ],
        {DialogueSystemRole.agent: 2, DialogueSystemRole.partner: 1},
    )

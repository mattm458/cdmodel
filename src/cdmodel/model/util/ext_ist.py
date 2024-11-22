from collections import OrderedDict

from torch import Tensor

from cdmodel.common.model import ISTSides, ISTStyle
from cdmodel.common.role_assignment import (
    AnalysisRole,
    DialogueSystemRole,
    Role,
    RoleType,
)
from cdmodel.model.components.ext_ist import ISTIncrementalEncoder, ISTOneShotEncoder


def ext_ist_validate(
    role_type: RoleType, ist_sides: ISTSides, ist_style: ISTStyle
) -> None:
    if (
        role_type == RoleType.DialogueSystem
        and ist_sides == ISTSides.single
        and ist_style == ISTStyle.one_shot
    ):
        pass
    elif (
        role_type == RoleType.DialogueSystem
        and ist_sides == ISTSides.both
        and ist_style == ISTStyle.one_shot
    ):
        pass
    elif (
        role_type == RoleType.DialogueSystem
        and ist_sides == ISTSides.both
        and ist_style == ISTStyle.blended
    ):
        pass
    elif (
        role_type == RoleType.Analysis
        and ist_sides == ISTSides.both
        and ist_style == ISTStyle.one_shot
    ):
        pass
    else:
        raise NotImplementedError(
            f"Unimplemented IST configuration {role_type} {ist_sides} {ist_style}"
        )


def ext_ist_one_shot_encode(
    encoder: ISTOneShotEncoder,
    role_type: RoleType,
    ist_sides: ISTSides,
    ist_style: ISTStyle,
    segment_features_delta_sides: dict[Role, Tensor],
    segment_features_delta_sides_len: dict[Role, list[int]],
    tokens: Tensor,
) -> tuple[OrderedDict[Role, Tensor], OrderedDict[Role, Tensor]]:
    out_embeddings: OrderedDict[Role, Tensor] = OrderedDict()
    out_weights: OrderedDict[Role, Tensor] = OrderedDict()

    if (
        role_type == RoleType.DialogueSystem
        and ist_sides == ISTSides.single
        and ist_style == ISTStyle.one_shot
    ):
        embeddings_agent, weights_agent = encoder(
            features=segment_features_delta_sides[DialogueSystemRole.agent],
            features_len=segment_features_delta_sides_len[DialogueSystemRole.agent],
            tokens=tokens,
        )
        out_embeddings[DialogueSystemRole.agent] = embeddings_agent
        out_weights[DialogueSystemRole.agent] = weights_agent
    elif (
        role_type == RoleType.DialogueSystem
        and ist_sides == ISTSides.both
        and ist_style == ISTStyle.one_shot
    ):
        embeddings_agent, weights_agent = encoder(
            features=segment_features_delta_sides[DialogueSystemRole.agent],
            features_len=segment_features_delta_sides_len[DialogueSystemRole.agent],
            tokens=tokens,
        )
        out_embeddings[DialogueSystemRole.agent] = embeddings_agent
        out_weights[DialogueSystemRole.agent] = weights_agent

        embeddings_partner, weights_partner = encoder(
            features=segment_features_delta_sides[DialogueSystemRole.partner],
            features_len=segment_features_delta_sides_len[DialogueSystemRole.partner],
            tokens=tokens,
        )
        out_embeddings[DialogueSystemRole.partner] = embeddings_partner
        out_weights[DialogueSystemRole.partner] = weights_partner
    elif (
        role_type == RoleType.DialogueSystem
        and ist_sides == ISTSides.both
        and ist_style == ISTStyle.blended
    ):
        embeddings_agent, weights_agent = encoder(
            features=segment_features_delta_sides[DialogueSystemRole.agent],
            features_len=segment_features_delta_sides_len[DialogueSystemRole.agent],
            tokens=tokens,
        )
        out_embeddings[DialogueSystemRole.agent] = embeddings_agent
        out_weights[DialogueSystemRole.agent] = weights_agent
    elif (
        role_type == RoleType.Analysis
        and ist_sides == ISTSides.both
        and ist_style == ISTStyle.one_shot
    ):
        embeddings_a, weights_a = encoder(
            features=segment_features_delta_sides[AnalysisRole.a],
            features_len=segment_features_delta_sides_len[AnalysisRole.a],
            tokens=tokens,
        )
        out_embeddings[AnalysisRole.a] = embeddings_a
        out_weights[AnalysisRole.a] = weights_a

        embeddings_b, weights_b = encoder(
            features=segment_features_delta_sides[AnalysisRole.b],
            features_len=segment_features_delta_sides_len[AnalysisRole.b],
            tokens=tokens,
        )
        out_embeddings[AnalysisRole.b] = embeddings_b
        out_weights[AnalysisRole.b] = weights_b
    else:
        raise NotImplementedError(
            f"Unimplemented IST configuration {role_type} {ist_sides} {ist_style}"
        )

    return out_embeddings, out_weights


def ext_ist_incremental_encode(
    encoder: ISTIncrementalEncoder,
    role_type: RoleType,
    ist_sides: ISTSides,
    ist_style: ISTStyle,
    speaker_role_idx: Tensor,
    features_delta: Tensor,
    accumulator: Tensor,
    h: Tensor,
    tokens: Tensor,
) -> tuple[Tensor, Tensor]:
    if (
        role_type == RoleType.DialogueSystem
        and ist_sides == ISTSides.both
        and ist_style == ISTStyle.blended
    ):
        mask = speaker_role_idx == DialogueSystemRole.agent.value
        return encoder(
            features=features_delta,
            h=h,
            accumulator=accumulator,
            mask=mask,
            tokens=tokens,
        )
    else:
        raise NotImplementedError(
            f"Unimplemented IST configuration {role_type} {ist_sides} {ist_style}"
        )

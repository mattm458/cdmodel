from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor

from cdmodel.common.model import ISTSides, ISTStyle
from cdmodel.common.role_assignment import (
    BothPredictionRole,
    AgentPartnerPredictionRole,
    Role,
    PredictionType,
)
from cdmodel.model.components.ext_ist import ISTIncrementalEncoder, ISTOneShotEncoder


def ext_ist_validate(
    role_type: PredictionType, ist_sides: ISTSides, ist_style: ISTStyle
) -> None:
    if (
        role_type == PredictionType.AgentPartner
        and ist_sides == ISTSides.single
        and ist_style == ISTStyle.one_shot
    ):
        pass
    elif (
        role_type == PredictionType.AgentPartner
        and ist_sides == ISTSides.both
        and ist_style == ISTStyle.one_shot
    ):
        pass
    elif (
        role_type == PredictionType.AgentPartner
        and ist_sides == ISTSides.both
        and ist_style == ISTStyle.blended
    ):
        pass
    elif (
        role_type == PredictionType.Both
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
    role_type: PredictionType,
    ist_sides: ISTSides,
    ist_style: ISTStyle,
    segment_features_sides: dict[Role, Tensor],
    segment_features_delta_sides: dict[Role, Tensor],
    segment_features_sides_len: dict[Role, list[int]],
    offset: Optional[Tensor] = None,
) -> tuple[OrderedDict[Role, Tensor], OrderedDict[Role, Tensor]]:
    out_embeddings: OrderedDict[Role, Tensor] = OrderedDict()
    out_weights: OrderedDict[Role, Tensor] = OrderedDict()

    if (
        role_type == PredictionType.AgentPartner
        and ist_sides == ISTSides.single
        and ist_style == ISTStyle.one_shot
    ):
        embeddings_agent, weights_agent = encoder(
            feature_deltas=segment_features_delta_sides[AgentPartnerPredictionRole.agent],
            feature_values=segment_features_sides[AgentPartnerPredictionRole.agent],
            features_len=segment_features_sides_len[AgentPartnerPredictionRole.agent],
            offset=offset,
        )
        out_embeddings[AgentPartnerPredictionRole.agent] = embeddings_agent
        out_weights[AgentPartnerPredictionRole.agent] = weights_agent
    elif (
        role_type == PredictionType.AgentPartner
        and ist_sides == ISTSides.both
        and ist_style == ISTStyle.one_shot
    ):
        embeddings_agent, weights_agent = encoder(
            feature_deltas=segment_features_delta_sides[AgentPartnerPredictionRole.agent],
            feature_values=segment_features_sides[AgentPartnerPredictionRole.agent],
            features_len=segment_features_sides_len[AgentPartnerPredictionRole.agent],
            offset=offset,
        )
        out_embeddings[AgentPartnerPredictionRole.agent] = embeddings_agent
        out_weights[AgentPartnerPredictionRole.agent] = weights_agent

        embeddings_partner, weights_partner = encoder(
            feature_deltas=segment_features_delta_sides[AgentPartnerPredictionRole.partner],
            feature_values=segment_features_sides[AgentPartnerPredictionRole.partner],
            features_len=segment_features_sides_len[AgentPartnerPredictionRole.partner],
            offset=offset,
        )
        out_embeddings[AgentPartnerPredictionRole.partner] = embeddings_partner
        out_weights[AgentPartnerPredictionRole.partner] = weights_partner
    elif (
        role_type == PredictionType.AgentPartner
        and ist_sides == ISTSides.both
        and ist_style == ISTStyle.blended
    ):
        embeddings_agent, weights_agent = encoder(
            feature_deltas=segment_features_delta_sides[AgentPartnerPredictionRole.agent],
            feature_values=segment_features_sides[AgentPartnerPredictionRole.agent],
            features_len=segment_features_sides_len[AgentPartnerPredictionRole.agent],
            offset=offset,
        )
        out_embeddings[AgentPartnerPredictionRole.agent] = embeddings_agent
        out_weights[AgentPartnerPredictionRole.agent] = weights_agent
    elif (
        role_type == PredictionType.Both
        and ist_sides == ISTSides.both
        and ist_style == ISTStyle.one_shot
    ):
        embeddings_a, weights_a = encoder(
            feature_deltas=segment_features_delta_sides[BothPredictionRole.a],
            feature_values=segment_features_sides[BothPredictionRole.a],
            features_len=segment_features_sides_len[BothPredictionRole.a],
            offset=offset,
        )
        out_embeddings[BothPredictionRole.a] = embeddings_a
        out_weights[BothPredictionRole.a] = weights_a

        embeddings_b, weights_b = encoder(
            feature_deltas=segment_features_delta_sides[BothPredictionRole.b],
            feature_values=segment_features_sides[BothPredictionRole.b],
            features_len=segment_features_sides_len[BothPredictionRole.b],
            offset=offset,
        )
        out_embeddings[BothPredictionRole.b] = embeddings_b
        out_weights[BothPredictionRole.b] = weights_b
    else:
        raise NotImplementedError(
            f"Unimplemented IST configuration {role_type} {ist_sides} {ist_style}"
        )

    return out_embeddings, out_weights


def ext_ist_incremental_encode(
    encoder: ISTIncrementalEncoder,
    role_type: PredictionType,
    ist_sides: ISTSides,
    ist_style: ISTStyle,
    speaker_role_idx: Tensor,
    features_delta: Tensor,
    accumulator: Tensor,
    h: Tensor,
) -> tuple[Tensor, Tensor]:
    if (
        role_type == PredictionType.AgentPartner
        and ist_sides == ISTSides.both
        and ist_style == ISTStyle.blended
    ):
        mask = speaker_role_idx == AgentPartnerPredictionRole.agent.value
        return encoder(
            features=features_delta,
            h=h,
            accumulator=accumulator,
            mask=mask,
        )
    else:
        raise NotImplementedError(
            f"Unimplemented IST configuration {role_type} {ist_sides} {ist_style}"
        )

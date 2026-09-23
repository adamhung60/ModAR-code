"""Balanced action/dynamics loss budget across schedule modes.

Each schedule mode aggregated its per-modality losses differently -- ``unified``
and ``independent`` summed them, the block-causal path averaged them, ``disjoint``
had a single active term -- and each mode supervises a different set of modalities
per stream. The two effects compounded, so the weight action carried in the
objective depended on the method: ~11% for RoboTwin ``unified``, 50% for
``modar``/``disjoint``. Methods trained under different objectives are not
comparable, which is the whole point of the sweep.

This module resolves an explicit per-source, per-modality loss coefficient so that
in expectation every mode spends the same fraction of its objective on action:

    total_share(action) = action_share            (0.5 by default)
    total_share(m)      = (1 - action_share) / n_dyn   for each dynamics modality

The realized share of a modality is not just its coefficient: a source only
contributes when it actually supervises that modality, which for some modes is a
per-step coin flip. So the resolver needs ``P(source, modality)`` -- the
probability that a source's forward supervises that modality -- and divides it out:

    c(s, m) = share(m) / ( P(s, m) * sum of w_s' over sources with P(s', m) > 0 )

That is exact by construction, since

    sum_s w_s * P(s, m) * c(s, m) = share(m).

One formula covers action and dynamics. ``P`` is keyed on ``(mode, stream)``
because different training formulations supervise different target sets.

Note what this does and does not equalize. The coefficient is the exact dial on a
modality's contribution to the gradient, so this fixes the shape of the objective.
It does not make action half the numeric loss value (dynamics losses have much
larger raw magnitudes), nor half the gradient norm. Cross-method comparability
wants the former: a single fixed objective, identical across arms.
"""
from __future__ import annotations

ACTION = "action"
STREAMS = ("joint", "action", "dynamics")

def dynamics_modalities(modalities) -> list:
    """The non-action modalities, in config order."""
    return [name for name in modalities if name != ACTION]


def normalize_stream(stream) -> str:
    """Canonical stream label. ``None`` is how the trainer spells ``joint``."""
    if stream is None:
        return "joint"
    stream = str(stream)
    if stream not in STREAMS:
        raise ValueError(f"unknown stream {stream!r} (expected one of {STREAMS})")
    return stream


def _disjoint_probs(modalities, candidates, p_of) -> dict:
    """Active-modality probabilities for one reduced disjoint pass.

    ``sample_disjoint`` renormalizes ``p_*`` over the candidates its stream
    allows, so the resolver must do the same.
    """
    if not candidates:
        raise ValueError("disjoint stream allows no modality to be active")
    weights = {name: (1.0 if p_of is None else float(p_of(name)))
               for name in candidates}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"disjoint p_* sum to {total} over {candidates}")
    return {name: weights.get(name, 0.0) / total for name in modalities}


def supervision_probs(mode, modalities, stream, p_of=None) -> dict:
    """P(modality is supervised) for one source, keyed on ``(mode, stream)``.

    Returns an entry for every modality in ``modalities``; 0.0 means the source
    never contributes gradient for it.
    """
    stream = normalize_stream(stream)
    dyn = dynamics_modalities(modalities)
    has_action = ACTION in modalities
    always = {name: 1.0 for name in modalities}
    never = {name: 0.0 for name in modalities}

    if mode == "action_only":
        return {**never, **({ACTION: 1.0} if has_action else {})}

    if mode == "disjoint":
        if stream == ACTION:
            candidates = [ACTION] if has_action else []
        elif stream == "dynamics":
            candidates = list(dyn)
        else:
            candidates = list(modalities)
        return _disjoint_probs(modalities, candidates, p_of)

    if stream == "dynamics":
        return {**never, **{name: 1.0 for name in dyn}}

    if stream == ACTION:
        # Every two-stream mode restricts the action source to action supervision.
        # In causal modes the dynamics prefix remains clean context; unified still
        # computes all co-noised outputs but masks their direct losses.
        return {**never, **({ACTION: 1.0} if has_action else {})}

    return dict(always)


def normalized_weights(sources) -> dict:
    """Source weights normalized to sum to one, keyed by source name."""
    raw = {}
    for source in sources:
        name = str(source["name"])
        if name in raw:
            raise ValueError(f"duplicate source name {name!r}")
        weight = float(source.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"source {name!r} weight must be > 0, got {weight}")
        raw[name] = weight
    if not raw:
        raise ValueError("at least one data source is required")
    total = sum(raw.values())
    return {name: weight / total for name, weight in raw.items()}


def resolve_loss_coeffs(mode, modalities, sources, action_share=0.5,
                        p_of=None) -> dict:
    """Per-source, per-modality loss coefficients hitting the target budget.

    ``sources`` are dicts with ``name``, ``stream`` (``None`` for joint), and an
    optional ``weight``. Returns ``{source_name: {modality: coefficient}}``,
    omitting modalities the source never supervises.

    ``action_share`` degrades gracefully: it collapses to 0 when no source carries
    action (a dynamics-only pretrain) and to 1 when none carries dynamics
    (``action_only``), so those runs keep a total coefficient mass of one.
    """
    modalities = tuple(modalities)
    dyn = dynamics_modalities(modalities)
    weights = normalized_weights(sources)
    probs = {str(source["name"]): supervision_probs(
        mode, modalities, source.get("stream"), p_of) for source in sources}

    # Weight of the sources that can supervise each modality at all.
    carrier_weight = {
        name: sum(weights[src] for src, prob in probs.items() if prob[name] > 0)
        for name in modalities}

    action_share = float(action_share)
    if not 0.0 <= action_share <= 1.0:
        raise ValueError(f"action_share must be in [0, 1], got {action_share}")
    if carrier_weight.get(ACTION, 0.0) <= 0:
        action_share = 0.0
    elif not any(carrier_weight[name] > 0 for name in dyn):
        action_share = 1.0

    share = {ACTION: action_share}
    live_dyn = [name for name in dyn if carrier_weight[name] > 0]
    for name in dyn:
        share[name] = (1.0 - action_share) / len(live_dyn) if live_dyn else 0.0

    coeffs = {}
    for name, prob in probs.items():
        coeffs[name] = {
            modality: share[modality] / (prob[modality] * carrier_weight[modality])
            for modality in modalities
            if prob[modality] > 0 and share[modality] > 0}
    return coeffs


def realized_shares(mode, modalities, sources, coeffs, p_of=None) -> dict:
    """Expected coefficient mass each modality receives per optimizer step.

    The inverse of :func:`resolve_loss_coeffs`, used for logging and tests.
    """
    weights = normalized_weights(sources)
    shares = {name: 0.0 for name in modalities}
    for source in sources:
        name = str(source["name"])
        prob = supervision_probs(mode, modalities, source.get("stream"), p_of)
        for modality, coeff in coeffs.get(name, {}).items():
            shares[modality] += weights[name] * prob[modality] * coeff
    return shares


def describe_loss_coeffs(mode, modalities, sources, coeffs, p_of=None) -> str:
    """Multi-line summary of a resolved budget, for the training log."""
    weights = normalized_weights(sources)
    shares = realized_shares(mode, modalities, sources, coeffs, p_of)
    header = f"mode={mode}"
    if ACTION in modalities:
        header += f" action share={shares[ACTION]:.3f}"
    lines = [header]
    for source in sources:
        name = str(source["name"])
        stream = normalize_stream(source.get("stream"))
        terms = " ".join(
            f"{modality}={coeff:.4g}"
            for modality, coeff in coeffs.get(name, {}).items())
        lines.append(f"  {name} (stream={stream} w={weights[name]:.3f}): "
                     f"{terms or 'no supervised modality'}")
    lines.append("  realized share " + " ".join(
        f"{modality}={shares[modality]:.4g}" for modality in modalities)
        + f" total={sum(shares.values()):.4g}")
    return "\n".join(lines)

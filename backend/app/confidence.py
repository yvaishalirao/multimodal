"""Composite per-field confidence with degraded-signal tracking (S3-T5, INV-6).

Score = weighted mean of the component signals that actually produced a
result. The weighting alone would let a missing signal quietly raise the
score (dropping a low component from the mean helps the average), so each
*required* component that failed to produce a result also marks the field
confidence_degraded -- and a degraded field always needs review, whatever
its number says.

Components:
- schema       (required) per-field validation: pass=1, fail/missing=0.
- consistency  (required when a rule applies to the field) any fail=0, all
               pass=1. A rule that errored is a failed component: degraded.
               No applicable rule means nothing to check: excluded, not
               degraded.
- ocr_layout   (required) Docling's parse confidence; None is degraded.
- input        (required) the extraction call had full multi-modal input;
               a text-only call (S3-T2) is degraded.
- dual_pass    (optional) agreement between two extraction passes, 0..1;
               excluded when not run.
"""
from pydantic import BaseModel

from app.consistency import RuleResult, results_for_field
from app.validation import FieldValidation

CONFIDENCE_THRESHOLD = 0.85

WEIGHTS = {"schema": 0.4, "consistency": 0.3, "ocr_layout": 0.2, "dual_pass": 0.1}


class FieldConfidence(BaseModel):
    field_name: str
    score: float | None
    components: dict[str, float]
    degraded: bool
    degraded_reasons: list[str]
    needs_review: bool
    review_reasons: list[str]


def _consistency_component(rules: list[RuleResult]) -> tuple[float | None, list[str]]:
    degraded = [f"consistency rule '{r.rule}' errored: {r.detail}" for r in rules if r.status == "error"]
    checked = [r for r in rules if r.status in ("pass", "fail")]
    if not checked:
        return None, degraded
    return (0.0 if any(r.status == "fail" for r in checked) else 1.0), degraded


def score_field(
    validation: FieldValidation,
    rule_results: list[RuleResult],
    parse_confidence: float | None,
    input_degraded_reasons: list[str],
    dual_pass_agreement: float | None = None,
) -> FieldConfidence:
    name = validation.field_name

    if validation.status == "missing" and not validation.required:
        # Nothing was extracted, so there is nothing to be confident in --
        # and an absent optional field (no tax line) isn't a review item.
        return FieldConfidence(
            field_name=name, score=None, components={}, degraded=False,
            degraded_reasons=[], needs_review=False, review_reasons=[],
        )

    components: dict[str, float] = {"schema": 1.0 if validation.status == "pass" else 0.0}
    degraded_reasons = list(input_degraded_reasons)

    consistency, consistency_degraded = _consistency_component(results_for_field(rule_results, name))
    degraded_reasons += consistency_degraded
    if consistency is not None:
        components["consistency"] = consistency

    if parse_confidence is None:
        degraded_reasons.append("ocr_layout confidence unavailable")
    else:
        components["ocr_layout"] = parse_confidence

    if dual_pass_agreement is not None:
        components["dual_pass"] = dual_pass_agreement

    weight = sum(WEIGHTS[c] for c in components)
    score = sum(WEIGHTS[c] * v for c, v in components.items()) / weight

    review_reasons = []
    if degraded_reasons:
        review_reasons.append("confidence degraded")
    if validation.status != "pass":
        review_reasons.append(f"validation {validation.status}")
    if score < CONFIDENCE_THRESHOLD:
        review_reasons.append(f"score {score:.2f} below {CONFIDENCE_THRESHOLD}")

    return FieldConfidence(
        field_name=name,
        score=round(score, 4),
        components=components,
        degraded=bool(degraded_reasons),
        degraded_reasons=degraded_reasons,
        needs_review=bool(review_reasons),
        review_reasons=review_reasons,
    )


def score_fields(
    validations: dict[str, FieldValidation],
    rule_results: list[RuleResult],
    parse_confidence: float | None,
    input_degraded_reasons: list[str],
    dual_pass_agreement: dict[str, float] | None = None,
) -> dict[str, FieldConfidence]:
    agreement = dual_pass_agreement or {}
    return {
        name: score_field(
            validation, rule_results, parse_confidence, input_degraded_reasons, agreement.get(name)
        )
        for name, validation in validations.items()
    }

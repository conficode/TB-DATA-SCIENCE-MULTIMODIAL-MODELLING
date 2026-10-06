"""Agreement-aware probability fusion.

WHY A RULE AND NOT A FITTED MODEL
The CNN was trained on chest X-ray images and the clinical model on a separate (synthetic) patient
table. No patient has BOTH an X-ray and clinical data, so there is no validation set on which fusion
weights could be estimated. Equal weights are therefore a transparent ASSUMPTION, not an optimum.
`scripts/evaluate_fusion.py` estimates weights once confirmed paired cases exist in the database.

RULE
1. Each model is first judged against ITS OWN validated threshold (CNN 0.55, clinical 0.147).
2. fused_probability = w_cnn * p_cnn + w_clin * p_clin      (weights 0.5 / 0.5)
   fused_threshold   = w_cnn * t_cnn + w_clin * t_clin      (so the fused cut-off is consistent
                                                             with the individual cut-offs)
3. Agreement: both positive / both negative = concordant; otherwise discordant.
4. Uncertainty: HIGH if discordant or |p_cnn - p_clin| >= 0.50; MODERATE if any probability is
   borderline (within 0.10 of its threshold); otherwise LOW.
5. Final assessment: discordant cases are NEVER resolved silently by the average. They are flagged
   'Indeterminate - clinician review' and, for screening safety, confirmatory testing is advised.
"""
from dataclasses import dataclass, asdict
import config as C


@dataclass
class FusionResult:
    cnn_probability: float
    clinical_probability: float
    cnn_positive: bool
    clinical_positive: bool
    fused_probability: float
    fused_threshold: float
    fused_positive: bool
    agreement: str            # "concordant_positive" | "concordant_negative" | "discordant"
    uncertainty: str          # "low" | "moderate" | "high"
    assessment: str
    recommendation: str
    reasons: list

    def to_dict(self):
        return asdict(self)


def fuse(p_cnn, p_clin, t_cnn=C.CNN_THRESHOLD, t_clin=0.147, weights=C.FUSION_WEIGHTS):
    w_c, w_k = weights["cnn"], weights["clinical"]
    fused = w_c * p_cnn + w_k * p_clin
    t_fused = w_c * t_cnn + w_k * t_clin
    cnn_pos, clin_pos = p_cnn >= t_cnn, p_clin >= t_clin
    reasons = []

    if cnn_pos and clin_pos:
        agreement = "concordant_positive"
    elif not cnn_pos and not clin_pos:
        agreement = "concordant_negative"
    else:
        agreement = "discordant"
        who = "X-ray model" if cnn_pos else "clinical model"
        reasons.append(f"Only the {who} is above its threshold - the models disagree.")

    gap = abs(p_cnn - p_clin)
    borderline = [n for n, p, t in (("X-ray", p_cnn, t_cnn), ("Clinical", p_clin, t_clin)) if abs(p - t) < C.BORDERLINE_MARGIN]
    if gap >= C.CONFLICT_GAP:
        reasons.append(f"Large probability gap between models ({gap:.2f}).")
    for n in borderline:
        reasons.append(f"{n} probability is close to its decision threshold (borderline).")

    if agreement == "discordant" or gap >= C.CONFLICT_GAP:
        uncertainty = "high"
    elif borderline:
        uncertainty = "moderate"
    else:
        uncertainty = "low"
        reasons.append("Both models agree and neither is near its threshold.")

    if agreement == "concordant_positive":
        assessment = "High suspicion of TB"
        rec = "Refer for confirmatory testing (e.g. GeneXpert MTB/RIF or sputum culture)."
    elif agreement == "concordant_negative":
        assessment = "Low suspicion of TB"
        rec = "TB not suggested by either model. Continue routine care; re-assess if symptoms persist."
    else:
        assessment = "Indeterminate - models disagree"
        rec = "Clinician review required. For screening safety, confirmatory testing is advised."

    return FusionResult(round(p_cnn, 4), round(p_clin, 4), bool(cnn_pos), bool(clin_pos), round(fused, 4),
                        round(t_fused, 4), bool(fused >= t_fused), agreement, uncertainty, assessment, rec, reasons)

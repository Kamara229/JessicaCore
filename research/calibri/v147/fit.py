"""Teacher-fit only on earlier designated source FIT, no holdout used.

A single shared int8 output-token bias approximates a global output logit
residual. This CANNOT recover contextual connections; audit actual top1 and
NLL separately rather than claiming teacher-aligned answers from scalar KL.
"""
import math
import numpy as np


def fit(mean_teacher_logits,mean_student_logits):
    t=np.asarray(mean_teacher_logits,dtype=np.float64)
    s=np.asarray(mean_student_logits,dtype=np.float64)
    if t.shape!=(49152,) or s.shape!=t.shape:
        raise ValueError("expected full original vocab logits")
    gap=t-s
    if not np.isfinite(gap).all():
        raise ValueError("nonfinite teacher-student output residual")
    # Softmax is invariant to a constant across all tokens.
    gap=gap-gap.mean()
    max_value=float(np.quantile(np.abs(gap),.995))
    if max_value<=0 or not math.isfinite(max_value):
        raise ValueError("no teacher correction signal")
    step=max_value/127
    code=np.rint(np.clip(gap/step,-127,127)).astype(np.int8)
    return code,step,{
        "mean_residual_centered":True,
        "predeclared_absolute_outlier_winsor_percentile":99.5,
        "fraction_int8_saturated":float(np.mean(np.abs(gap/step)>127)),
        "unique_int8_bias_values":len(np.unique(code)),
        "mean_abs_fit_teacher_student_logit_gap":float(np.mean(np.abs(gap))),
        "post_int8_quantization_RMSE":float(
            np.sqrt(np.mean((gap-code.astype(np.float64)*step)**2))),
    }

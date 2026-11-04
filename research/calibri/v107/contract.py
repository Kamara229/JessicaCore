"""SHA-pinned completed outputs NOT seen by v106 posthoc original v96 source.

This is a cross-run observational check, not prospective blinded QA and
not the independent v89 TEST. Outputs were partially seen in earlier
research, but these threshold constants came from v96 alone.
"""
from __future__ import annotations
SOURCE_FILES={
    "v103":{
        "name":"v103_same_byte_head_candidate_chat_ground_truth_and_visible_text.json",
        "sha256":"ef784535ab611177a0dfdf13ce0dacd9b6df65b5388c877340320194d9b2b383",
        "source_run":41747347901,
        "original_FP32":"original_FP32",
        "compressed_models":("original_CB59","same_byte_v102"),
        "expected_prompts":4,
    },
    "v104":{
        "name":"v104_actual_original_head_bank_source_real_CPU_and_semantic_tests.json",
        "sha256":"ab6ee4c70a40f5384cd8ee0851fdd2bce369efef4bbed2628e9124f42b430df5",
        "source_run":41748235592,
        "original_FP32":"FP32",
        "compressed_models":("CB59","CB104_banked"),
        "expected_prompts":2,
    }
}
assert SOURCE_FILES["v103"]["source_run"]!=SOURCE_FILES["v104"]["source_run"]

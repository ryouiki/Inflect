from __future__ import annotations

from inflect_finetune.training import (
    STAGE_ADAPT,
    STAGE_DECODER,
    STAGE_POSTERIOR,
    TrainingOptions,
    _stage_for_step,
)


def _options() -> TrainingOptions:
    return TrainingOptions(
        base_model="nano",
        prepared_dir="prepared",
        output_dir="run",
        posterior_warmup_steps=10,
        decoder_unfreeze_step=20,
    )


def test_stage_boundaries_select_the_stage_for_the_next_step() -> None:
    options = _options()

    assert _stage_for_step(options, 0) == STAGE_POSTERIOR
    assert _stage_for_step(options, 9) == STAGE_POSTERIOR
    assert _stage_for_step(options, 10) == STAGE_ADAPT
    assert _stage_for_step(options, 19) == STAGE_ADAPT
    assert _stage_for_step(options, 20) == STAGE_DECODER

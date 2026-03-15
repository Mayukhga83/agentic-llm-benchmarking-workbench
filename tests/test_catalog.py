from core.model_catalog import (
    candidate_config_ids,
    default_candidate_config_ids,
    get_candidate_config,
    get_preset,
)


def test_exactly_eight_candidate_configurations() -> None:
    assert len(candidate_config_ids()) == 8


def test_three_default_candidates_are_valid() -> None:
    defaults = default_candidate_config_ids()
    assert len(defaults) == 3
    assert set(defaults).issubset(set(candidate_config_ids()))


def test_only_two_presets_and_strict_uses_gpt_54_mini() -> None:
    strict = get_preset("Strict")
    balanced = get_preset("Balanced")
    assert strict["evaluator"]["model"] == "gpt-5.4-mini"
    assert strict["judge"]["model"] == "gpt-5.4-mini"
    assert balanced["evaluator"]["model"] == "gpt-4o-mini"
    assert balanced["judge"]["model"] == "gpt-4o-mini"


def test_candidate_specs_have_prices() -> None:
    for config_id in candidate_config_ids():
        spec = get_candidate_config(config_id)
        assert spec["input_price_per_million"] >= 0
        assert spec["output_price_per_million"] >= 0

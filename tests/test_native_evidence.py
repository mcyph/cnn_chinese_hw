import numpy as np
import pytest
import torch

from iso_tools.inference.types import ClassifierEvidence, ClassifierModel, ModelIdentity, Vocabulary
from cnn_chinese_hw.recognizer.evidence import project_handwriting
from cnn_chinese_hw.recognizer.calibration import split_calibration_samples


def test_native_projection_matches_union_temperature_ensemble_and_missing_support():
    vocabularies = [("日", "目", "本"), ("目", "木", "日")]
    logits = [np.asarray([2.0, -1.5, 0.1], dtype=np.float32), np.asarray([1.1, 0.5, 2.3], dtype=np.float32)]
    temperatures = [0.8, 1.7]
    evidence = ClassifierEvidence(tuple(ClassifierModel(ModelIdentity(str(i), "test"), Vocabulary(tokens),
        f"logits_{i}", temperatures[i]) for i, tokens in enumerate(vocabularies)), (), ("no_tta",))
    result = project_handwriting(evidence, {f"logits_{i}": values for i, values in enumerate(logits)})
    expected = dict.fromkeys(("日", "目", "本", "木"), 0.0)
    for tokens, values, temperature in zip(vocabularies, logits, temperatures):
        for token, probability in zip(tokens, torch.softmax(torch.from_numpy(values) / temperature, dim=0)):
            expected[token] += float(probability) / 2
    assert [x.text for x in result.candidates] == sorted(expected, key=expected.get, reverse=True)
    assert {x.text: x.probability for x in result.candidates} == pytest.approx(expected, abs=1e-7)
    missing = next(x for x in result.candidates if x.text == "木").model_probabilities[0]
    assert missing.support == "not_in_vocabulary" and missing.probability is None
    assert result.common_vocabulary_size == 2
    assert result.common_vocabulary_js_nats > 0
    assert result.out_of_vocabulary_detection == "unsupported"


def test_handwriting_scores_and_disagreement_preserve_truncated_mass():
    models = tuple(ClassifierModel(ModelIdentity(str(i), "test"), Vocabulary(("日", "目")), f"x{i}") for i in range(2))
    evidence = ClassifierEvidence(models, (), ())
    result = project_handwriting(evidence, {"x0": [0, 0], "x1": [0, 0]}, limit=1)
    assert result.candidates[0].text == "日"  # Deterministic tie, not a calibrated winner.
    assert result.omitted_mass == 0.5
    assert result.top_margin == 0
    assert result.common_vocabulary_js_nats == 0
    with pytest.raises(ValueError, match="finite"):
        project_handwriting(evidence, {"x0": [np.nan, 0], "x1": [0, 0]})


def test_selection_calibration_test_separate_character_variants_and_training_duplicates():
    rows = [(label, [[(label * 10, 0), (label * 10 + 1, 1)]], False) for label in range(20)]
    rows += [rows[3], rows[7]]
    result = split_calibration_samples([rows[0]], rows, seed=4)
    assert result.excluded_training_duplicates == (0,)
    groups = [set(indexes) for indexes in (result.selection, result.calibration, result.test)]
    assert all(groups) and not any(a & b for i, a in enumerate(groups) for b in groups[i + 1:])
    labels = [{rows[i][0] for i in indexes} for indexes in groups]
    assert not any(a & b for i, a in enumerate(labels) for b in labels[i + 1:])
    assert set.union(*groups) == set(range(1, len(rows)))
    assert result.unavailable_identities == ("writer_id", "device_id")
    reordered = split_calibration_samples([rows[0]], list(reversed(rows)), seed=4)
    assert reordered.partition_sha256 == result.partition_sha256
    with pytest.raises(ValueError, match="too few"):
        split_calibration_samples([], rows[:2])

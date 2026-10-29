"""Focused regressions for the local NOGA pilot traversal."""
from noga_pilot.run import section_rotations, top2_paths, top2_trigger


def test_division_uncertainty_triggers_without_section_uncertainty():
    section = {"choice": "A", "probabilities": {"A": .9, "B": .1}}
    division = {"choice": "A1", "probabilities": {"A1": .52, "A2": .48}}
    assert top2_trigger(section, division)


def test_two_uncertain_sections_and_divisions_are_pruned_to_two_ranked_paths():
    children = {"A": ["A1", "A2"], "B": ["B1", "B2"]}
    for division in ("A1", "A2", "B1", "B2"):
        children[division] = [f"{division}g1", f"{division}g2"]
        for group in children[division]:
            children[group] = [f"{group}c1", f"{group}c2"]
            for klass in children[group]:
                children[klass] = [f"{klass}t1", f"{klass}t2"]
    calls = []

    def infer(uid, arm, level, purpose, lineage, candidates):
        calls.append((level, tuple(code for _, code in lineage)))
        if level == "division":
            lead = .57 if lineage[0][1] == "B" else .54
            probs = {candidates[0]: lead, candidates[1]: 1 - lead}
        else:
            probs = {candidates[0]: .9}
        return {"choice": candidates[0], "probabilities": probs,
                "answer_confidence": max(probs.values()), "error": None}

    found = top2_paths("u", "p", {"choice": "A", "probabilities": {"A": .51, "B": .49}},
                       children, infer)
    assert found == ["B1g1c1t1", "A1g1c1t1"]
    assert len(calls) == 8
    assert len(calls) <= 12
    assert len({lineage[0] for level, lineage in calls if level == "division"}) == 2


def test_single_child_levels_need_no_calls_and_failed_branch_is_dropped():
    children = {"A": ["A1"], "A1": ["A1g"], "A1g": ["A1c"], "A1c": ["A1t"],
                "B": ["B1", "B2"]}
    calls = []

    def infer(uid, arm, level, purpose, lineage, candidates):
        calls.append(level)
        return {"choice": None, "probabilities": {}, "error": "injected"}

    assert top2_paths("u", "p", {"choice": "A", "probabilities": {"A": .51, "B": .49}},
                      children, infer) == ["A1t"]
    assert calls == ["division"]


def test_error_rotation_is_recorded_and_later_rotation_continues():
    calls = 0

    def predict(_purpose, _question, key):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected")
        return {"choice": "A", "probabilities": {"A": 1.0}, "answer_confidence": 1.0}

    rotations = list(section_rotations(predict, "purpose", {"A": "Alpha", "B": "Beta"},
                                       {"A": "01", "B": "02"}))
    assert len(rotations) == 2
    assert "injected" in rotations[0][1]["error"]
    assert rotations[1][1]["choice"] == "01"
    assert rotations[-1][3]

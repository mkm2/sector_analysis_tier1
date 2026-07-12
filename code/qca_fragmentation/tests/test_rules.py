"""Rule-encoding tests (context Tier 1 sec.1.2) + HSF numbering map."""
from qca_fragmentation.core import rules


VERIFY = {
    150: ("I", "V", "V", "I"),
    156: ("I", "I", "V", "I"),
    22:  ("I", "V", "V", "D"),
    204: ("I", "I", "I", "I"),
    51:  ("V", "V", "V", "V"),
    201: ("V", "I", "I", "I"),
    0:   ("D", "D", "D", "D"),
}


def test_verification_cases():
    for r, exp in VERIFY.items():
        assert rules.wolfram_to_tuple(r) == exp
        assert rules.tuple_to_wolfram(exp) == r


def test_bijection_256():
    seen = set()
    for r in range(256):
        t = rules.wolfram_to_tuple(r)
        assert rules.tuple_to_wolfram(t) == r
        seen.add(t)
    assert len(seen) == 256


def test_reflection_swaps_r01_r10():
    # rule 156 = (I,I,V,I) -> reflect swaps r01,r10 -> (I,V,I,I)
    assert rules.reflect_tuple(("I", "I", "V", "I")) == ("I", "V", "I", "I")
    # reflection is an involution
    for r in range(256):
        assert rules.reflect_wolfram(rules.reflect_wolfram(r)) == r


def test_unitary_set_is_16():
    assert len(rules.UNITARY_RULES) == 16
    assert all(rules.is_unitary(rules.wolfram_to_tuple(r)) for r in rules.UNITARY_RULES)


def test_hsf_map():
    # sector notes: HSF rule -> Wolfram number
    expected = {0: 204, 1: 201, 2: 198, 3: 195, 4: 156, 5: 153, 6: 150, 7: 147,
                8: 108, 9: 105, 10: 102, 11: 99, 12: 60, 13: 57, 14: 54, 15: 51}
    assert rules.HSF_TO_WOLFRAM == expected
    for h, w in expected.items():
        assert rules.wolfram_to_hsf(w) == h


def test_spinflip_only_vfree():
    # rule 204 (I,I,I,I) is V-free: spin-flip should succeed
    rules.spinflip_wolfram(204)
    # rule 150 contains V: must refuse
    try:
        rules.spinflip_wolfram(150)
        assert False
    except ValueError:
        pass

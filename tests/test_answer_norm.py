"""Cases the two graders disagreed on, pinned so the protocol cannot silently drift back.

Each case here is a real disagreement between our reconstructed judge and the pinned
Qwen3-32B on the same 526 PhysReason problems, plus the physics conventions those
disagreements turned on. A rule that cannot get these right is not worth shipping.
"""

import unittest

from pairaudit.scoring.answer_norm import compare, compare_any, parse


class TestUnitRule(unittest.TestCase):
    """A bare prediction matches a gold with units; the unit is fixed by the problem."""

    def test_bare_number_matches_gold_with_unit(self):
        # our judge YES, open judge NO -- our judge was right
        self.assertIs(compare("6V", "6"), True)
        self.assertIs(compare(r"$4\mathrm{m}/\mathrm{s}$", "4"), True)
        self.assertIs(compare("20 m/s", "20"), True)

    def test_matching_units_match(self):
        self.assertIs(compare("6V", "6 V"), True)
        self.assertIs(compare("0.025m", r"0.025\,\mathrm{m}"), True)

    def test_unit_conversion(self):
        self.assertIs(compare("0.025m", "2.5 cm"), True)
        self.assertIs(compare("1 km", "1000 m"), True)
        self.assertIs(compare("2 hours", "7200 s"), True)

    def test_incompatible_units_do_not_match(self):
        self.assertIs(compare("5V", "5A"), False)
        self.assertIs(compare("10 m", "10 s"), False)


class TestSignAndDirection(unittest.TestCase):
    """Sign is significant; a missing direction is tolerated, a contradictory one is not."""

    def test_sign_error_rejected(self):
        # our judge YES, open judge NO -- the OPEN judge was right
        self.assertIs(compare("0.025m", "-0.025"), False)
        self.assertIs(compare("6V", "-6"), False)

    def test_missing_direction_tolerated(self):
        self.assertIs(compare(r"$1\mathrm{m}$, west", "1"), True)

    def test_contradictory_direction_rejected(self):
        self.assertIs(compare("5 m/s east", "5 m/s west"), False)
        self.assertIs(compare("3 m upward", "3 m downward"), False)

    def test_negative_gold_matches_negative_pred(self):
        self.assertIs(compare("-1200V", "-1200"), True)


class TestNumericForms(unittest.TestCase):
    def test_scientific_notation(self):
        self.assertIs(compare(r"1.26 \times 10^{-7}", "1.26e-7"), True)
        self.assertIs(compare("4.36e6", r"4.4 \times 10^{6}"), True)   # 1% tolerance

    def test_fractions(self):
        self.assertIs(compare(r"\frac{8}{3}", "2.67"), True)
        self.assertIs(compare("22/3", "7.33"), True)

    def test_close_but_distinct_rejected(self):
        # the campaign's stated rule: ~3% apart is NO, even though it looks close
        self.assertIs(compare("1.68 m/s", "1.66"), False)
        self.assertIs(compare("180 N", "172.8"), False)


class TestDefersRatherThanGuesses(unittest.TestCase):
    """The protocol must return None, not a guess, wherever it cannot decide."""

    def test_symbolic_gold_defers(self):
        self.assertIsNone(compare(r"\frac{mg}{k}", "2.5"))
        self.assertIsNone(compare(r"\sqrt{2gh}", "4.4"))

    def test_multi_valued_gold_defers(self):
        self.assertIsNone(compare("0.05s or 0.15s", "0.05"))
        self.assertIsNone(compare("100V, -300V, -200V", "100"))

    def test_unknown_unit_defers(self):
        self.assertIsNone(compare("5 widgets", "5"))

    def test_empty_defers(self):
        self.assertIsNone(compare("", "5"))
        self.assertIsNone(compare("6V", ""))


class TestCompareAny(unittest.TestCase):
    """`any one of the boxed answers may match` -- but only over decidable ones."""

    def test_any_match_wins(self):
        self.assertIs(compare_any("6V", ["2", "6", "9"]), True)

    def test_all_decidable_and_wrong(self):
        self.assertIs(compare_any("6V", ["2", "9"]), False)

    def test_no_decidable_predictions_defers(self):
        self.assertIsNone(compare_any("6V", [r"\frac{mg}{k}"]))

    def test_empty_prediction_list_defers(self):
        # an empty list is "no answer", which the caller scores NO -- not our call
        self.assertIsNone(compare_any("6V", []))


class TestParse(unittest.TestCase):
    def test_canonicalises_to_base_units(self):
        a = parse("2.5 cm")
        self.assertEqual(a.unit, "m")
        self.assertAlmostEqual(a.value, 0.025)

    def test_extracts_direction(self):
        a = parse("5 m/s east")
        self.assertEqual(a.direction, "east")
        self.assertAlmostEqual(a.value, 5.0)


if __name__ == "__main__":
    unittest.main()

"""Behavioural tests for the frozen matcher (carried over unchanged from the research repo)."""

import json
import time
import unittest
from pathlib import Path

from pairaudit.scoring.answer_equiv import equivalent

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "physcorp_gold_samples.json"


class AnswerEquivalenceTests(unittest.TestCase):
    def test_numeric_golds_use_one_percent_relative_tolerance_after_unit_strip(self):
        self.assertTrue(equivalent("0.615 kg", "0.61"))
        self.assertTrue(equivalent(r"0.615\,\mathrm{kg}", "0.61"))
        self.assertTrue(equivalent("66.7 N", r"\frac{200}{3}"))
        self.assertTrue(equivalent("200/3", r"\frac{200}{3}"))
        self.assertFalse(equivalent("0.7 kg", "0.61"))
        self.assertFalse(equivalent("0.61 apples", "0.61"))

    def test_mismatched_units_on_both_sides_fail_closed(self):
        """Same mantissa, different unit must NOT match — wrong-SI-prefix answers
        are a common model failure and rewarding them teaches wrong units."""
        self.assertFalse(equivalent("5 cm", "5 mm"))
        self.assertFalse(equivalent("2.5 m", "2.5 km"))
        self.assertFalse(equivalent("3 eV", "3 MeV"))
        self.assertFalse(equivalent("10 ms", "10 ns"))
        self.assertFalse(equivalent(r"1.1 \times 10^{6} eV", "1.1e6 MeV"))
        self.assertTrue(equivalent("5 mm", "5 mm"))
        self.assertTrue(equivalent(r"2\,\mathrm{kg}", "2 kg"))
        self.assertTrue(equivalent("3 m/s", "3 m / s"))
        self.assertFalse(equivalent("3 m/s", "3 m*s"))

    def test_scientific_times_notation_is_numeric(self):
        self.assertTrue(equivalent("1.1e6", r"1.10 \times 10^{6}"))
        self.assertTrue(equivalent(r"7.16\times10^{-3}", "0.00716"))
        self.assertFalse(equivalent("1.1e5", r"1.10 \times 10^{6}"))

    def test_fraction_roots_subscripts_and_implicit_products_are_symbolic(self):
        gold = r"\frac{m_{1} m_{2} g}{m_{1} + m_{2}}"
        self.assertTrue(equivalent("m_1*m_2*g/(m_1+m_2)", gold))
        self.assertTrue(equivalent(r"\frac{\sqrt{2}}{2}", r"1/\sqrt{2}"))
        self.assertTrue(equivalent("alpha^2 + theta", "α^2 + θ"))
        self.assertFalse(equivalent("m_1*m_2/(m_1+m_2)", gold))

    def test_equation_lhs_can_be_retained_reversed_or_omitted(self):
        self.assertTrue(equivalent(r"\sigma T^4", r"I(T) = \sigma T^{4}"))
        self.assertTrue(equivalent(r"D_N=R_0+D_0", r"D_0+R_0=D_N"))
        self.assertTrue(
            equivalent(
                r"\kappa(p,T)=\zeta^2/(1-\zeta^2)",
                r"\kappa(p, T) = \frac{\zeta^{2}}{1-\zeta^{2}}",
            )
        )
        self.assertTrue(equivalent("1.6e-7", r"\mu \approx 1.6 \times 10^{-7}"))
        self.assertFalse(equivalent(r"\sigma T^3", r"I(T) = \sigma T^{4}"))

    def test_latex_function_powers_parse_as_function_application(self):
        gold = r"I(\theta)=b^{2} \frac{\sin ^{2} v}{v^{2}}"
        self.assertTrue(equivalent("b^2*(sin(v)/v)^2", gold))
        self.assertFalse(equivalent("b^2*sin(v)/v^2", gold))

    def test_unknown_indexed_functions_are_stable_symbolic_atoms(self):
        gold = r"\overline{\xi(r)} \cdot \frac{\hbar^{2}}{2}(2l+1)"
        self.assertTrue(equivalent("xi(r)*hbar^2*(2*l+1)/2", gold))
        self.assertTrue(equivalent("10^-16*N(0)", r"10^{-16} N(0)"))
        self.assertFalse(equivalent("10^-16*N(1)", r"10^{-16} N(0)"))
        self.assertFalse(equivalent("0", r"10^{-16} N(0)"))

    def test_multi_quantity_answers_compare_componentwise(self):
        gold = r"-\frac{1}{10 \sqrt{5}}, \frac{1}{10 \sqrt{5}}, 0, 0"
        self.assertTrue(
            equivalent(r"-\sqrt{5}/50, \sqrt{5}/50, 0, 0", gold)
        )
        self.assertFalse(
            equivalent(r"\sqrt{5}/50, -\sqrt{5}/50, 0, 0", gold)
        )

    def test_near_exact_symbolic_coefficients_use_deterministic_sampling(self):
        self.assertTrue(equivalent("0.333333333333*x", "x/3"))
        self.assertFalse(equivalent("0.34*x", "x/3"))

    def test_malformed_and_unsafe_expressions_fail_closed(self):
        self.assertFalse(equivalent("", "x"))
        self.assertFalse(equivalent("sqrt(", "x"))
        self.assertFalse(equivalent("sqrt(", "sqrt("))
        self.assertFalse(equivalent("__import__('os')", "x"))
        self.assertFalse(equivalent("__class__", "__class__"))
        self.assertFalse(equivalent("x", "__class__"))

    def test_real_fixture_has_five_complete_buckets_and_reflexive_fallback(self):
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            set(fixture),
            {"numeric", "times_notation", "equation_lhs", "unit_suffixed", "symbolic"},
        )
        for bucket, golds in fixture.items():
            self.assertEqual(len(golds), 12, bucket)
            for gold in golds:
                self.assertTrue(equivalent(gold, gold), f"{bucket}: {gold}")

    def test_semantic_variants_from_every_real_fixture_bucket(self):
        cases = (
            ("5.03", "5.0295"),
            ("1.1e6", r"1.10 \times 10^{6}"),
            (
                "p_0*V_0/(2*p_0+x*rho*g)",
                r"V_x = \frac{p_0 V_0}{2p_0 + x\rho g}",
            ),
            (
                "acos(1-4*m_1^2*l_1/((m_1+m_2)^2*l_2)*(1-cos(alpha)))",
                r"\alpha_{2}=\arccos\left[1-"
                r"\frac{4m_{1}^{2}l_{1}}{(m_{1}+m_{2})^{2}l_{2}}"
                r"(1-\cos\alpha)\right]",
            ),
            ("t*v_0", r"v_0 t"),
            (
                "alpha^4*m_e*c^2/hbar",
                r"\alpha^{4} \frac{m_{\mathrm{e}} c^{2}}{\hbar}",
            ),
            (
                "-N*k*T*(1-log(y)+0.177*y+0.001650*y^2)",
                r"-N k T\left(1-\ln y+0.177 y+1.650 \times 10^{-3} y^{2}\right)",
            ),
        )
        for prediction, gold in cases:
            self.assertTrue(equivalent(prediction, gold), gold)

    def test_hand_checked_symbolic_negatives_do_not_match(self):
        negatives = (
            ("alpha^3*m_e*c^2/hbar", r"\alpha^{4} \frac{m_e c^2}{\hbar}"),
            ("t+v_0", r"v_0 t"),
            ("p_0*V_0/(p_0+x*rho*g)", r"\frac{p_0 V_0}{2p_0+x\rho g}"),
            ("sin(v)/v", r"\frac{\sin^2 v}{v^2}"),
        )
        for prediction, gold in negatives:
            self.assertFalse(equivalent(prediction, gold), gold)

    def test_pathological_symbolic_input_is_wall_clock_bounded(self):
        pathological = "+".join("x" for _ in range(300_000))
        started = time.monotonic()
        self.assertFalse(equivalent(pathological, "x"))
        self.assertLess(time.monotonic() - started, 3.0)

    # --- regression guards for the 2026-08-31 matcher fixes -------------------
    #
    # b6103a8 ($-delimiter stripping) and 97d31ff (\dfrac/\tfrac/\cfrac aliasing)
    # both shipped with NO test. The consequence was not hypothetical: the full
    # suite passed identically with and without them, so nothing could tell a
    # fixed checkout from a stale one. A probe then ran on a stale checkout, and
    # the defect was differential between its arms -- an arm shown the gold copies
    # the gold's notation and slips past a formatting-sensitive matcher, while the
    # unhinted arm writes the same physics differently and is scored wrong. That
    # probe's headline result had to be retracted (see
    # probes/setup_injection/THREE_ARM_HINT.md).
    #
    # These two assertions are the exact canary pair used by the provenance guard
    # in probes/setup_injection/three_arm_hint_probe.py. Both are False on any
    # tree predating the fixes and True at HEAD, so they are what makes a stale
    # matcher detectable at all. Do not weaken them without replacing the guard.

    def test_dollar_delimiters_are_stripped_before_unit_parsing(self):
        """b6103a8. PhysReason golds are $-wrapped; predictions usually are not."""
        self.assertTrue(equivalent("2", r"$2\mathrm{m}/\mathrm{s}$"))
        self.assertTrue(equivalent("0.61", "$0.615 kg$"))
        self.assertTrue(equivalent(r"$$9.8$$", "9.8"))
        # Stripping must not make disagreeing values agree.
        self.assertFalse(equivalent("3", r"$2\mathrm{m}/\mathrm{s}$"))

    def test_unbraced_fraction_arguments_follow_tex(self):
        """TeX reads ``\\frac12`` as 1/2: single tokens are arguments. A fraction with no
        argument at all is malformed and fails closed through _normalized_string."""
        from pairaudit.scoring.answer_equiv import _latex_to_infix, _normalized_string

        self.assertEqual(_latex_to_infix(r"\dfrac12"), "((1)/(2))")
        self.assertEqual(_latex_to_infix(r"\tfrac ab"), "((a)/(b))")
        self.assertTrue(equivalent(r"\cfrac12", "0.5"))
        for text in (r"\dfrac", r"\dfrac x"):
            _normalized_string(text)   # must not raise
        self.assertEqual(_normalized_string(r"\dfrac12 mgh"), "((1)/(2))mgh")
        self.assertEqual(_normalized_string(r"\dfrac12 m g h"), "((1)/(2))mgh")

    def test_dfrac_variants_alias_to_frac(self):
        r"""97d31ff. \dfrac appears in 17.5% of predictions and 0.2% of golds."""
        self.assertTrue(equivalent(r"\dfrac{1}{2}", r"\frac{1}{2}"))
        self.assertTrue(equivalent(r"\tfrac{1}{2}", "0.5"))
        self.assertTrue(equivalent(r"\cfrac{200}{3}", "66.7"))
        # Aliasing must not collapse distinct fractions.
        self.assertFalse(equivalent(r"\dfrac{1}{3}", r"\frac{1}{2}"))


class ScoringDefinitionTests(unittest.TestCase):
    """design/SCORING_DEFINITION.md D1-D7, pinned on the acceptance sample's own misses."""

    def test_glued_products_are_products_of_single_letter_symbols(self):
        # The largest miss class: "mg" parsed as one symbol named mg, so spacing decided
        # equivalence. cal_problem_00346#0 and friends.
        self.assertTrue(equivalent(r"\frac{mg\sqrt{3}}{q}", r"$\frac{\sqrt{3}mg}{q}$"))
        self.assertTrue(equivalent("mgh", "m g h"))
        self.assertTrue(equivalent(r"G\frac{Mm}{r^2}", r"\frac{G M m}{r^{2}}"))
        self.assertTrue(equivalent(r"E = v_0 B", r"v_{0}B"))
        self.assertTrue(equivalent(r"G \cos \alpha - F \sin \beta", r"Gcos\alpha - Fsin\beta"))
        self.assertTrue(equivalent(r"m v_0^2 / r + mg - kQq/r^2",
                                   r"$m\frac{v_{0}^{2}}{r} + m g - \frac{k Q q}{r^{2}}$"))
        self.assertFalse(equivalent("mgh", "m g"))
        self.assertTrue(equivalent(r"\frac{GMm}{R}", r"\frac{GM}{R}m"))
        # Greek and function names survive whole; var-forms are the same symbol.
        self.assertTrue(equivalent(r"\theta", "theta"))
        self.assertTrue(equivalent(r"\sin\theta", r"\sin \theta"))
        self.assertTrue(equivalent(r"\frac{q}{4\pi\varepsilon_0 r}", r"\frac{q}{4 \pi \epsilon_{0} r}"))
        self.assertTrue(equivalent(r"v_1 = \frac{Q_1}{4\pi r^2}", r"$v_1 = \frac{Q_1}{4\pir^2}$"))
        # Identifiers glued to a leading digit, and constants times a letter that is a
        # symbol rather than a unit (corpus losses under the first rewrite).
        self.assertTrue(equivalent(r"\dfrac{2}{3}mgh", r"$\frac{2mgh}{3}$"))
        self.assertTrue(equivalent(r"3\sqrt{\dfrac{2mL}{qE}}", r"$3\sqrt{\frac{2m L}{q E}}$"))
        self.assertTrue(equivalent(r"\frac{5T}{3}", r"$\frac{5}{3}T$"))
        self.assertTrue(equivalent(r"\dfrac{9L}{2}", "$4.5L$"))
        self.assertTrue(equivalent(r"\dfrac{m}{2}", r"$\frac{1}{2}m"))
        self.assertTrue(equivalent(r"\sqrt{2} \, mg", r"$\sqrt{2}m g$"))
        self.assertTrue(equivalent(r"\dfrac{7}{2}mgR", "$3.5mgR$"))
        self.assertFalse(equivalent(r"\dfrac{7}{2}mgR", "$3.5mg$"))
        # A stray backslash on a symbol keeps its subscript; \ell is l.
        self.assertTrue(equivalent(r"m \frac{v_0^2}{r} + mg - \frac{k Q q}{r^2}",
                                   r"$m\frac{\v_{0}^{2}}{r} + m g - \frac{k Q q}{r^{2}}$"))
        self.assertTrue(equivalent(r"\frac{mgl}{2}", r"\frac{mg\ell}{2}"))

    def test_symbols_are_positive_physical_quantities(self):
        self.assertTrue(equivalent(r"\sqrt{\dfrac{2 h v_0^2 R}{x^2 - h^2}}",
                                   r"$v_{0}{\sqrt{\frac{2h R}{x^{2}-h^{2}}}}$"))
        self.assertTrue(equivalent(r"\sqrt{v^2}", "v"))
        self.assertFalse(equivalent(r"-mgl\tan\alpha", r"$mgl\tan \alpha$"))
        self.assertFalse(equivalent(r"\sqrt{v_0^2 - \frac{2eEd}{m}}",
                                    r"$v_{s}=\sqrt{v_{0}^{2}-\frac{4e E d}{m}}$"))
        self.assertFalse(equivalent(r"\frac{q}{4\pi\varepsilon_0 r}", r"\frac{q}{4 \pi \epsilon_{0} r^2}"))

    def test_d1_units_compare_by_dimension_not_string(self):
        self.assertTrue(equivalent("3 N/C", "3 V/m"))
        self.assertTrue(equivalent(r"3 m \cdot s^{-1}", "3 m/s"))
        self.assertTrue(equivalent(r"25", r"25\mathrm{kg \cdot m/s}"))
        self.assertTrue(equivalent(r"4.6", r"$4.6\mathrm{m}/\mathrm{s}^{2}$"))
        self.assertTrue(equivalent(r"0.3\,\text{N}", "0.30N"))
        self.assertTrue(equivalent("5.6", "5.6Wb"))
        self.assertTrue(equivalent("2 J", r"2 N \cdot m"))
        self.assertFalse(equivalent(r"4.6 \, \text{m/s}", r"$4.6\mathrm{m}/\mathrm{s}^{2}$"))
        self.assertFalse(equivalent("3 m/s", "3 m*s"))
        self.assertFalse(equivalent("2 h", "2 m"))
        # Conversion is out of scope: prefixed units stay opaque and fail closed.
        self.assertFalse(equivalent(r"1.001 \text{ A}", "1005mA"))
        self.assertFalse(equivalent(r"1.5 \text{ GHz}", r"$1.5 \times 10^9$"))

    def test_d2_constants_roots_exponentials_and_primes(self):
        self.assertTrue(equivalent(r"2\sqrt{5}", r"$2\sqrt5$ m/s"))
        self.assertTrue(equivalent(r"\pi^2", r"\pi^{2} W"))
        self.assertTrue(equivalent(r"h = \left( \dfrac{GM T^2}{4\pi^2} \right)^{1/3} - R",
                                   r"$h=\sqrt[3]{\frac{G M T^{2}}{4\pi^{2}}}-R$"))
        self.assertTrue(equivalent(r"\exp(-t/\tau)", r"e^{-t/\tau}"))
        self.assertTrue(equivalent(r"\frac{e^2}{4\pi\varepsilon_0 r}",
                                   r"\frac{e^{2}}{4 \pi \varepsilon_{0} r}"))
        self.assertTrue(equivalent(r"T - \frac{a' N}{3 k_B V}",
                                   r"$T^{\prime}=T-\frac{a^{\prime} N}{3 k_{B} V}$"))
        self.assertTrue(equivalent(r"\dfrac{\lambda L}{8d}", r"$y=\frac{\lambda L}{8 d}\mathrm{m}$"))
        self.assertFalse(equivalent(r"\dfrac{\lambda L}{8d}\,\mathrm{s}", r"$y=\frac{\lambda L}{8 d}\mathrm{m}$"))

    def test_degrees_percent_and_ratios_are_numeric(self):
        self.assertTrue(equivalent("45", r"$45^{\circ}$"))
        self.assertTrue(equivalent(r"45^\circ", "45°"))
        self.assertFalse(equivalent(r"60^\circ", r"$45^{\circ}$"))
        self.assertTrue(equivalent(r"60\%", r"$\eta=60%$"))
        self.assertFalse(equivalent("0.6", r"$\eta=60%$"))
        self.assertTrue(equivalent("1:50", "1:50"))
        self.assertTrue(equivalent("0.02", "1:50"))
        self.assertFalse(equivalent("50", "1:50"))

    def test_d5_label_prefixes_name_the_quantity(self):
        self.assertTrue(equivalent("6", r"$v=6\mathrm{m}/\mathrm{s}$"))
        self.assertTrue(equivalent("0.1", "$k = 0.1$"))
        self.assertTrue(equivalent("10", r"$v_{F} = 10{m}/{s}$"))
        self.assertFalse(equivalent("31.2", r"$v_{F} = 10{m}/{s}$"))
        self.assertTrue(equivalent(r"\text{Sub-question 2: } \frac{21}{20} g", r"\frac{21g}{20}"))

    def test_d3_multi_value_golds_credit_the_matched_fraction(self):
        from pairaudit.scoring.answer_equiv import match_fraction

        self.assertEqual(match_fraction(["250", "150"], "$250N$, $150N$"), 1.0)
        self.assertEqual(match_fraction([r"0.3\,\text{N}", r"0.06\,\text{N}"], "0.30N, 0.06N"), 1.0)
        self.assertEqual(
            match_fraction(["0.4", r"\dfrac{8\sqrt{3}}{15}"], r"0.4m, \frac{8\sqrt{3}}{15} m"), 1.0
        )
        self.assertEqual(match_fraction(["36"], r"36s, 90\sqrt{5}m"), 0.5)
        self.assertEqual(match_fraction(["4"], "$Z=4$, Beryllium(Be)"), 0.5)
        self.assertEqual(match_fraction(["36, 90\\sqrt{5}"], r"36s, 90\sqrt{5}m"), 1.0)
        self.assertEqual(match_fraction(["7"], r"36s, 90\sqrt{5}m"), 0.0)
        self.assertEqual(match_fraction(["2", "3"], "2"), 1.0)
        self.assertEqual(match_fraction(["3"], "2"), 0.0)
        self.assertEqual(match_fraction([], "2"), 0.0)

    def test_d4_d7_gold_scorability(self):
        from pairaudit.scoring.answer_equiv import gold_scorability

        self.assertEqual(gold_scorability(r"$18\mathrm{N}$, direction horizontally to the left"), "prose")
        self.assertEqual(gold_scorability("remains constant."), "prose")
        self.assertEqual(gold_scorability("The distance is 14m, and the displacement is 10m."), "prose")
        self.assertEqual(gold_scorability("$Z=4$, Beryllium(Be)"), "symbolic")
        self.assertEqual(gold_scorability("1005mA"), "numeric")
        self.assertEqual(gold_scorability(r"$45^{\circ}$"), "numeric")
        self.assertEqual(gold_scorability(r"\frac{mg}{q}"), "symbolic")
        self.assertEqual(gold_scorability(r"$v=6\mathrm{m}/\mathrm{s}$"), "numeric")
        self.assertEqual(gold_scorability(r"$B<\frac{4}{3}$ or $2<B$"), "unparseable")
        self.assertEqual(gold_scorability(""), "unparseable")
        self.assertEqual(gold_scorability(r"25\mathrm{kg \cdot m/s}"), "numeric")


if __name__ == "__main__":
    unittest.main()


class SampledRatioTests(unittest.TestCase):
    """_sampled_ratio: the leak filter's rational-multiple primitive."""

    def _ratio(self, a, b):
        from pairaudit.scoring.answer_equiv import _parse_symbolic, _sampled_ratio

        return _sampled_ratio(_parse_symbolic(a), _parse_symbolic(b))

    def test_constant_ratio_detected(self):
        self.assertAlmostEqual(self._ratio("m*v^2", "m*v^2/2"), 2.0, places=6)
        self.assertAlmostEqual(self._ratio("g*h", "2*g*h"), 0.5, places=6)
        self.assertAlmostEqual(self._ratio("3*x*y/4", "x*y"), 0.75, places=6)

    def test_non_constant_ratio_returns_none(self):
        self.assertIsNone(self._ratio("x^2", "x"))
        self.assertIsNone(self._ratio("x + y", "x*y"))

    def test_degenerate_denominator_survives(self):
        # x - x is identically 0: every draw hits the denominator guard.
        self.assertIsNone(self._ratio("x*y", "x - x"))

    def test_sampled_equivalent_untouched(self):
        from pairaudit.scoring.answer_equiv import _parse_symbolic, _sampled_equivalent

        self.assertTrue(
            _sampled_equivalent(
                _parse_symbolic("0.333333333333*x"), _parse_symbolic("x/3")
            )
        )
        self.assertFalse(
            _sampled_equivalent(_parse_symbolic("0.34*x"), _parse_symbolic("x/3"))
        )

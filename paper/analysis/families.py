"""Released base -> RL pairs read in the paper, in table order. ADDED: the six pre-registered on 2026-10-11 (expansion/PREREGISTRATION.md).
RECIPES env selects a subset in the analysis scripts: "original" (first eight), "added" (the six), "amend1" (the four of amendment 1),
"added10" (six + four), "base:<base name>" (one base's recipes), unset = all."""
import os
FAMILIES = {
    "DeepSeek-R1-Distill-Qwen-1.5B": ["DeepScaleR-1.5B-Preview", "Nemotron-Research-Reasoning-Qwen-1.5B", "Open-RS3", "STILL-3-1.5B-preview",
                                      "JustRL-DeepSeek-1.5B", "DeepMath-1.5B", "RLinf-math-1.5B", "AReaL-1.5B-Preview-Stage-3"],
    "DeepSeek-R1-Distill-Qwen-7B": ["Skywork-OR1-7B", "AceReason-Nemotron-7B", "Skywork-OR1-Math-7B"],
    "Qwen3-4B": ["Polaris-4B-Preview"],
    "OpenMath-Nemotron-1.5B": ["JustRL-Nemotron-1.5B", "QuestA-Nemotron-1.5B", "KnowRL-Nemotron-1.5B", "DeepMath-Omn-1.5B"],
    "Qwen3-1.7B": ["e3-1.7B", "Polaris-1.7B-Preview"],
}
ADDED = ["JustRL-Nemotron-1.5B", "QuestA-Nemotron-1.5B", "KnowRL-Nemotron-1.5B", "DeepMath-Omn-1.5B", "e3-1.7B", "Polaris-1.7B-Preview"]
AMEND1 = ["JustRL-DeepSeek-1.5B", "DeepMath-1.5B", "RLinf-math-1.5B", "AReaL-1.5B-Preview-Stage-3"]   # amendment 1, same day, before any output
def selected():
    sub = os.environ.get("RECIPES", "")
    if sub.startswith("base:"):   # exploratory within-base reads (not part of the registered analysis)
        return {sub[5:]: list(FAMILIES[sub[5:]])}
    keep = {"added": lambda r: r in ADDED, "amend1": lambda r: r in AMEND1, "added10": lambda r: r in ADDED + AMEND1,
            "original": lambda r: r not in ADDED + AMEND1}.get(sub, lambda r: True)
    return {b: [r for r in rs if keep(r)] for b, rs in FAMILIES.items() if any(keep(r) for r in rs)}
def perm_p(x, y, rho, n_mc=100000):
    """two-sided permutation p for Spearman's rho: exact up to 8 recipes, Monte Carlo beyond (9! permutations and up)."""
    import itertools, numpy as np
    from scipy import stats
    n = len(x)
    if n <= 8:
        vals = [stats.spearmanr(x, [y[i] for i in p]).statistic for p in itertools.permutations(range(n))]; how = "exact"
    else:
        g = np.random.default_rng(1); vals = [stats.spearmanr(x, [y[i] for i in g.permutation(n)]).statistic for _ in range(n_mc)]; how = f"monte carlo {n_mc}"
    return float(np.mean([abs(v) >= abs(rho) - 1e-12 for v in vals])), how

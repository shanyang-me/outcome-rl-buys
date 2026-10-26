# Paper tables: where each comes from and what is verified

Every table ships as CSV in `paper/tables/`. Status as of 2026-10-25:

- **rebuilt**: `pytest -m slow` reruns the paper's script, unchanged, on `data/rows` and `data/extra`, and its output matches
  the paper's result file to 1e-9 (JSON) or the printed table cell by cell (CSV).
- **rows here**: the rows are in `data/rows`, but no test reruns the table yet.
- **controlled run**: the 8B training runs (Section 5) are not part of this release.

| table (CSV) | what | script(s) in `paper/analysis` | status | test |
|---|---|---|---|---|
| capcells | every cap and beyond-cap cell, physics and MATH-500 | cap_split.py, cells_table.py | rebuilt (JSON) | test_paper_results |
| splitsens | pooled shares under three splits | fig1_ci.py, sens_table.py | rebuilt (fig1_ci.json) | test_paper_results |
| (Section 4, App. A shares) | pooled shares, base-resampled intervals | pooled_shares.py on capcells.csv | rebuilt (output text) | test_paper_results |
| hardcells, filter | harder physics set, judged and matcher | hard_reach.py, hard_by_source.py, hard_table.py | rebuilt (JSON and CSV) | test_paper_extended |
| hard32 | 24-sample round on the harder set | hard32_analysis.py, hard32_table.py | rebuilt (CSV) | test_paper_extended |
| mathcells, mathyears, mathharness | competition-math set | hard_reach.py, math_p4.py, ../math/p5_harness.py, math_table.py | rebuilt (CSV) | test_paper_extended |
| math32 | 24-sample round on competition math | hard32_analysis.py, hard32_table.py | rebuilt (CSV) | test_paper_extended |
| temp | authors' temperatures, eight pairs | temp_cells.py, temp_table.py | rebuilt (JSON per pair) | test_paper_extended |
| regrade, regrade2 | matcher vs judge, first and second generation | regrade_cells.py, regrade2_table.py | rebuilt (JSON; regrade2 also CSV) | test_paper_extended |
| reach32 | reach at k=32 | reach32_table.py | rebuilt (JSON and CSV) | test_paper_extended |
| publicpairs | every cell at 4k/8k/16k | pairs_ci.py, pairs_tables18.py | rows here | |
| pairsmain | one row per pair | stats_mine.py, pairs_main18.py | rows here | |
| hump | gain by base solve rate | hump_split.py | rows here | |
| pair | Qwen2.5-7B vs Open-Reasoner-Zero-7B | (pair_decomp run) | rows here | |
| robustreach | robust reach at the cap | split_pairs.py, reach_split.py | rows here; reading verdicts in results/ | |
| tracereading | labelled reading of RL-only wins | trace_stats.py, trace_tables.py | rows here; reading verdicts in results/ | |
| deadset, latent, reliance, seeds, trajectory, transfer, twinschan, protocols | controlled run | (research repository) | controlled run | |
| inprogress | reads added after the first version | (prose) | n/a | |

Notes:
- The two `hard32_analysis.json` copies in `paper/results` predate the last 24-sample rows and the Polaris-1.7B extension to all
  671 problems; the printed hard32 table (448 RL wins read, 289 base wins) matches the rerun, so the test checks the table.
- `temp_table.py` prints four at-cap rows twice; the paper keeps one of each. All 76 printed rows match the rerun.
- The competition-math result files were not kept; those tables are checked against the printed CSVs.

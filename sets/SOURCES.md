# Evaluation-set sources

`hard671.jsonl` and `math276.jsonl` ship ids, golds and `q_sha256` (sha256 of
the exact UTF-8 problem text the models saw) but no third-party problem text.
`build_sets.py` downloads each source below at the pinned revision, rebuilds
the text, checks every hash exactly, and writes `sets/build/{hard671,math276}.jsonl`.

```
pip install datasets huggingface_hub
python sets/build_sets.py
```

The script exits nonzero if any id is missing from its source or its rebuilt
text does not hash to `q_sha256`. Golds come from our sets files, not the source.

Ids have the form `<src>|<key>`. "Transform" is the exact function from the
source row to the text `q` that was hashed.

Licences are quoted from each dataset card at the pinned revision. They cover
the dataset as published on Hugging Face; the underlying problems may carry the
original authors' or competition organisers' rights.

## hard671 (671 problems)

| src | n | dataset | revision |
|---|---|---|---|
| OB-phys-en | 234 | Hothan/OlympiadBench | 91184b52131e7fc9455fef848035173aea8cc01a |
| UGPhysics-en-1part | 169 | UGPhysics/ugphysics | 523e71bc4e33356bb19de4af38c128796e8a8770 |
| PHYSICS-test | 118 | desimfj/PHYSICS | 6479757368c7d620ab826ba3ff7af44cb7eb3b5d |
| PHYBench | 100 | Eureka-Lab/PHYBench | d6d91c787b7abb865eb2490a328bf85a9f5095f0 |
| FrontierSci-phys-short | 50 | openai/frontierscience | 25ed67db7da8f4591484e764008ff585544f5a30 |

### OB-phys-en (OlympiadBench, physics, English, open-ended, text-only)
- URL: https://huggingface.co/datasets/Hothan/OlympiadBench/tree/91184b52131e7fc9455fef848035173aea8cc01a
- File: `OlympiadBench/OE_TO_physics_en_COMP/OE_TO_physics_en_COMP.parquet` (236 rows; we use 234)
- Key: the row's `id` (e.g. `OB-phys-en|802` is `id == 802`).
- Transform: `context + "\n\n" + question` when `context` is non-empty, else `question`.
  233 of our 234 rows have a context; one has an empty context and uses `question` alone.
- Licence (card YAML): `license: apache-2.0`

### UGPhysics-en-1part (UGPhysics, English)
- URL: https://huggingface.co/datasets/UGPhysics/ugphysics/tree/523e71bc4e33356bb19de4af38c128796e8a8770
- Files: `<Subject>/en.jsonl` for all 13 subject folders.
- Key: `<subject>_<index>`, where `subject` is the row's `subject` field (with
  spaces) and `index` its `index` field, e.g. `Quantum Mechanics_4576`.
- Transform: none (`problem` verbatim).
- Licence (card YAML): `license: cc-by-nc-sa-4.0`

### PHYSICS-test (PHYSICS benchmark, test split)
- URL: https://huggingface.co/datasets/desimfj/PHYSICS/tree/6479757368c7d620ab826ba3ff7af44cb7eb3b5d
- File: `data/test.jsonl` (2000 rows: each problem appears once with `language == "zh"`
  and once with `language == "en"` under the same `id`).
- Key: the row's `id`, restricted to `language == "en"`.
- Transform: none (`question` verbatim).
- Licence (card YAML): `license: mit`

### PHYBench
- URL: https://huggingface.co/datasets/Eureka-Lab/PHYBench/tree/d6d91c787b7abb865eb2490a328bf85a9f5095f0
- File: `PHYBench-fullques_v1.json` (the 100 problems released with solutions and answers).
- Key: the row's `id`.
- Transform: none (`content` verbatim).
- Licence (card YAML): `license: mit`; the card body also shows a "License: MIT" badge.

### FrontierSci-phys-short (FrontierScience, olympiad track, physics)
- URL: https://huggingface.co/datasets/openai/frontierscience/tree/25ed67db7da8f4591484e764008ff585544f5a30
- File: `olympiad/test.jsonl`, rows with `subject == "physics"` (50 rows).
- Key: the row's `task_group_id` (a uuid).
- Transform: none (`problem` verbatim).
- Licence (card YAML): `license: apache-2.0`

## math276 (276 problems, MathArena final-answer sets)

All eight come from MathArena dataset repos, file `data/train-00000-of-00001.parquet`.

- Key: `problem_idx` (e.g. `aime_2026|1` is `problem_idx == 1` in `MathArena/aime_2026`).
- Transform: `problem.strip()`. Some rows carry a leading or trailing newline in
  the source; the models saw the stripped text. Typos in the source (e.g.
  `\overlien` in `hmmt_nov_2025|1`) are kept as-is.
- Licence (each card, YAML `license: cc-by-nc-sa-4.0`, and its "Licensing
  Information" section): "This dataset is licensed under the
  Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
  Please abide by the license when using the provided data."

| src | n | dataset | revision | URL |
|---|---|---|---|---|
| aime_2025 | 30 | MathArena/aime_2025 | c94da77eb22bbd6439e62a323bec18493a421302 | https://huggingface.co/datasets/MathArena/aime_2025/tree/c94da77eb22bbd6439e62a323bec18493a421302 |
| aime_2026 | 30 | MathArena/aime_2026 | d61f486805306e6e6b5913504c07763a99e1b112 | https://huggingface.co/datasets/MathArena/aime_2026/tree/d61f486805306e6e6b5913504c07763a99e1b112 |
| hmmt_feb_2025 | 30 | MathArena/hmmt_feb_2025 | 6fdc4277120810ff75aa22d2d5489b91f7a262a1 | https://huggingface.co/datasets/MathArena/hmmt_feb_2025/tree/6fdc4277120810ff75aa22d2d5489b91f7a262a1 |
| hmmt_feb_2026 | 33 | MathArena/hmmt_feb_2026 | 589a53fe152febc0f4f99f3e561007bf8da8e664 | https://huggingface.co/datasets/MathArena/hmmt_feb_2026/tree/589a53fe152febc0f4f99f3e561007bf8da8e664 |
| hmmt_nov_2025 | 30 | MathArena/hmmt_nov_2025 | 118dbfb45c4c9467c672268ed55166642897aa46 | https://huggingface.co/datasets/MathArena/hmmt_nov_2025/tree/118dbfb45c4c9467c672268ed55166642897aa46 |
| brumo_2025 | 30 | MathArena/brumo_2025 | 6de9bc0987f5710c6376ee980e1483dbeb55cfda | https://huggingface.co/datasets/MathArena/brumo_2025/tree/6de9bc0987f5710c6376ee980e1483dbeb55cfda |
| cmimc_2025 | 40 | MathArena/cmimc_2025 | 7b3ca7c4b417d31b1f97f781f39fc4f97f1b9185 | https://huggingface.co/datasets/MathArena/cmimc_2025/tree/7b3ca7c4b417d31b1f97f781f39fc4f97f1b9185 |
| smt_2025 | 53 | MathArena/smt_2025 | d8e99d83f1583842f46d4953cd960eea5d887e45 | https://huggingface.co/datasets/MathArena/smt_2025/tree/d8e99d83f1583842f46d4953cd960eea5d887e45 |

## Shipping note

`sets/build/` holds rebuilt third-party text and is gitignored, like
`sets/_texts_local/`. UGPhysics and all MathArena sets are CC BY-NC-SA 4.0,
which is the main reason the release ships hashes rather than text.

## phys300 (texts shipped)

`sets/phys300.jsonl` ships the problem text `q` with each gold, because these 300 problems were drawn from the
research corpus and no public builder reproduces them. Each line names its source:

| source | n | licence |
|---|---|---|
| Physics-R1 corpus, derived from UGPhysics (https://huggingface.co/datasets/UGPhysics/ugphysics) | 150 | CC BY-NC-SA 4.0 |
| MegaScience/TextbookReasoning (https://huggingface.co/datasets/MegaScience/TextbookReasoning) | 99 | CC BY-NC-SA 4.0 |
| desimfj/PHYSICS (https://huggingface.co/datasets/desimfj/PHYSICS) | 32 | MIT |
| zhibei1204/PhysReason (https://huggingface.co/datasets/zhibei1204/PhysReason) | 19 | MIT |

Since 249 of the 300 texts are CC BY-NC-SA 4.0, `sets/phys300.jsonl` as a whole is distributed under CC BY-NC-SA 4.0
(attribution as above, non-commercial use, same licence for derivatives), not under the repository's Apache-2.0.

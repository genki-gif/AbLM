# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AbLM is a pipeline that downloads human paired antibody sequences from OAS (Observed Antibody Space), generates embeddings using three antibody language models (AntiBERTy, AbLang, AbLang-2), and produces UMAP visualizations and Pgen-PLL correlation analyses. The research goal is comparing how different models structure the antibody embedding space, with V-gene family (IGHV1–7) coloring.

## Environment Setup

```bash
# Create environment (one-time)
mamba env create -f environment.yml

# Run scripts
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/<script.py>
```

**Critical constraints:**
- `KMP_DUPLICATE_LIB_OK=TRUE` is required due to PyTorch/OpenMP conflicts
- `transformers` is pinned to `4.46.3` — AntiBERTy 0.1.3 is incompatible with v5+

## Pipeline Execution

```bash
# Step 1: Download and preprocess sequences (~8626 sequences from OAS)
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/01_download_data.py --n 10000

# Step 2: Generate embeddings for all 4 model variants
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/02_generate_embeddings.py --batch 32
# --model flag accepts: all | antiberty | ablang | ablang2 | ablang2heavy

# Step 3: Compute Pgen (OLGA) and PLL scores
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/03_compute_pgen_pll.py --batch 32
# --model flag accepts: all | pgen | antiberty | ablang | ablang2 | ablang2heavy

# Step 4: Visualize in notebook
# Select kernel "Python (ablm)" and run all cells
```

Scripts 2 and 3 have **caching**: if `.npy` output files already exist, they are skipped. Delete them manually to regenerate.

## Architecture

### Data Flow

```
OAS API → 01_download_data.py → data/sequences_processed.csv (8626 × 202)
                                         ↓
                          02_generate_embeddings.py → 8 .npy embedding files
                          [AntiBERTy(512d), AbLang(768d), AbLang-2 paired(480d), AbLang-2 heavy(480d)]
                          [× sequence-level and CDR-H3-level pooling]
                                         ↓
                          03_compute_pgen_pll.py → 10 .npy score files + pgen_pll_scores.csv
                          [OLGA Pgen + per-model PLL (full-seq and CDR3-specific)]
                                         ↓
                          notebook.ipynb → figures/ (UMAP plots + Pgen-PLL scatterplots)
```

### Key CSV Columns (`data/sequences_processed.csv`)

| Column | Content |
|--------|---------|
| `vh_seq` | Heavy chain VH amino acid sequence |
| `vl_seq` | Light chain VL amino acid sequence |
| `v_family` | V-gene family (e.g., `IGHV1`) |
| `v_call_heavy`, `j_call_heavy` | V/J allele names (used by OLGA) |
| `cdr3_aa_heavy` | CDR-H3 sequence (OAS format: excludes conserved C and W) |
| `cdr3_indices` | `(start, end)` indices of CDR-H3 in `vh_seq` |

### Embedding Design

All models receive the **full-length sequence** as input; CDR-H3 embeddings are extracted by slicing the output residue vectors (not by inputting CDR-H3 alone).

- `*_seq_vecs.npy`: mean-pool over all residues → (N, D)
- `*_cdr3_vecs.npy`: mean-pool over CDR-H3 residues → (N, D), falls back to full-sequence mean if indices invalid

### AbLang-2 Tokenization (Important)

AbLang-2 must use `w_extra_tkns=True`. Token layout:
```
Paired:      [<(0), VH[0..L-1](1..L_vh), >(L_vh+1), |(L_vh+2), <, VL..., >]
Heavy-only:  [<(0), VH[0..L-1](1..L_vh), >(L_vh+1), |]
```
VH residues are **always at indices 1 to 1+len(VH)**. For heavy-only input, pass `vl=""` to maintain the same token structure.

> **Past bug**: The old code used `w_extra_tkns=False` with `f"{vh}|{vl}"` format, causing the first VH residue to be missed and the `|` separator to be included in the extraction. Current code is fixed.

### OLGA CDR3 Format Conversion

OAS `cdr3_aa_heavy` excludes the conserved C (N-terminal) and W/F (C-terminal). OLGA requires both:
```python
cdr3_olga = 'C' + cdr3_aa_heavy + vh_seq[cdr3_end]
```

## Editing `notebook.ipynb`

Edit notebook cells by manipulating the JSON directly rather than using notebook tools:

```python
import json
with open('notebook.ipynb') as f:
    nb = json.load(f)
nb['cells'][i]['source'] = "new code"
with open('notebook.ipynb', 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

## Adding a New Model to the Pipeline

1. Add a `run_<model>()` function in `02_generate_embeddings.py` (reuse `pool_seq` / `pool_cdr3` utilities)
2. Add it to the `models_to_run` dispatch in `main()`
3. Update the `--model` help text
4. Update `notebook.ipynb`: add to `vectors` dict and `MODEL_NAMES` list

## Regenerating Data from Scratch

```bash
# Change sequence count or force full re-run
rm data/sequences_processed.csv data/*.npy
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/01_download_data.py --n <N>
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/02_generate_embeddings.py

# Re-run Pgen/PLL only
rm data/pgen.npy data/pgen_vj.npy data/*_pll.npy data/pgen_pll_scores.csv
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/03_compute_pgen_pll.py
```

## Known Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `OMP: Error #15` | OpenMP multi-load on macOS | Set `KMP_DUPLICATE_LIB_OK=TRUE` |
| `AttributeError: 'AntiBERTy'...all_tied_weights_keys` | transformers v5 incompatibility | `pip install "transformers==4.46.3"` |
| OAS API 404 | Old `/api/pairednumbers/` endpoint removed | Current code crawls `ngsdb/paired/` directly |
| AbLang weights download hanging | `requests.get()` hangs on large files | Download manually with `curl` and extract |
| Many `Pgen = 0` values | CDR3 terminal residue not W/F, or V/J allele not in OLGA | Normal — recorded as `NaN`, auto-excluded on log10 transform |

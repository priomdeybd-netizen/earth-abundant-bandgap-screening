# Two-stage ML screening of earth-abundant semiconductor absorbers

Code and results for the manuscript "Two-Stage Machine-Learning Screening of Earth-Abundant Semiconductor Absorbers from Composition: Accuracy, Limits Across Anion Families, and Window Probabilities".

Authors: Priom Dey (corresponding author, 2023131044@student.sust.edu), Avro Talukder, Yuvraj Roy. Department of Chemistry, Shahjalal University of Science and Technology, Sylhet, Bangladesh.

## Run order (Google Colab, free; each notebook is self-contained: Runtime > Run all)
1. `notebooks/step2_screening.ipynb` - two-stage model, three test protocols, candidate list from Materials Project.
2. `notebooks/step3_validation.ipynb` - chemical-system results with confidence intervals, per-family results, literature validation (upload `literature_validation.csv` when asked).
3. `notebooks/step4_calibration.ipynb` - calibration and enrichment of the window probability, seed stability, baseline regressors.
4. `make_paper_final.py` - the script that generated the first draft of `paper.tex` from the result folders. The final manuscript in `paper/` was edited by hand afterwards (authors, references, corrections), so it is the authoritative version.

## Contents
- `results/` - outputs reported in the paper (step 2, 3 and 4 tables, figures, `summary.json`).
- `literature_validation.csv` - input to step 3: all compiled literature rows with inclusion status.
- `expanded_literature_validation.csv` - the 12 compounds in Table 11 with predictions. The 12-compound literature comparison in the paper is in this file; `results/step3_results` is the earlier 4-compound run.
- `literature_verification_sheet.csv` - record of the source check for each literature value.
- `paper/` - final `paper.tex`, `paper.pdf` and figures.

## Data
Public Matbench `matbench_expt_gap` and `matbench_mp_e_form` data sets, loaded through `matminer`.

## Notes
- Element abundance values in the paper follow the CRC Handbook (97th ed.).
- The window-probability screen is a coarse filter; see the paper for its limits (oxides, unseen anion families).

## License
MIT (see `LICENSE`).

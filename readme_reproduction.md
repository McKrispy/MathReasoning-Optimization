# Reproduction instructions

## 1. What to run

Run commands below from this repository's root. Reproduction starts from the supplied checkpoint and requires **no retraining, external dataset, pretrained model download or API key**. Dependency installation may use the network; prediction and scoring work offline after installation.

The submitted method is **B17/B20 historical three-view Soft Stacking**. The exact implementation is `frozen/src/inference.py`, imported by the editable official interface `code/submission.py`. Supporting inference modules are under `frozen/src/features/`, `frozen/src/models/` and `frozen/src/postprocessing/`. They load only the nine files in `frozen/model_bundle/` and the inputs passed by the official runner.

- Arm A: embeddings-only cross-candidate attention, mean of seeds 42, 43 and 44.
- Arm B: embeddings plus 16 structured features, mean of the same three seeds.
- R5: one HistGradientBoosting tree model on 16 raw structured features plus a 64-dimensional SVD projection and its 64-dimensional question-centered version.
- Question-standardized view logits use weights `[0.6519943983184848, 0.2773940600630942, 0.07061154161842091]`, followed by the frozen validity/conflict masks and answer-cluster aggregation at temperature `0.5`.

The checkpoint includes learned means/scales, the SVD matrix, all six attention members, the tree and configuration. Neither the strict-refit alternative nor the later decoder, centroid or energy experiments replace this method. Do not average experimental checkpoints or refit/calibrate anything to reproduce the reported score.

`frozen/manifest.json` gives the actual release source/asset hashes and conservative asset accounting. The implementation's mathematical computation and the nine asset bytes are retained from the evaluated method; package-relative import paths and the public wrapper were adapted for this repository.

## 2. Directory roles

```text
code/                     Official data and tools; only submission.py is edited
guide/                    Original assignment guide
frozen/src/               Active frozen inference implementation
frozen/model_bundle/      The only inference checkpoint to use
frozen/requirements.txt   Runtime versions and active transitive dependencies
frozen/results/           Published scores/predictions and evaluation evidence
frozen/reproduce.py       Fresh-process reproduction and result comparison
development/src/          Historical development/analysis source archive
development/tests/        Historical development tests
development/runs/         Experiment results, predictions, weights and audits
development/requirements.txt  Additional training/test dependencies
development/publication_manifest.json  Archive copy hashes and normalization note
Final_Report.md           Empty report placeholder; not a finished report
```

The original `code/README.md` and `guide/GUIDE.md` describe the official protocol. The original package manifest still records the original empty submission template. That single template is intentionally replaced; `frozen/manifest.json` records the current wrapper. Official `predict.py`, `common.py`, `score_predictions.py`, `check_data.py`, supplied data and the remaining official files are unchanged.

## 3. Environment

The measured environment is **Windows, Python 3.12.0**, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1 and PyTorch 2.7.1+cu126. Full runtime pins are in `frozen/requirements.txt`. Preserve the exact scikit-learn version because the R5 estimator is serialized as a pickle. Only load the trusted supplied checkpoint.

Create and activate a Python 3.12 environment. For example, on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "torch==2.7.1+cu126" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

The wheel source follows the [official PyTorch 2.7.1 CUDA 12.6 installation instructions](https://pytorch.org/get-started/previous-versions/#v271). The CUDA-tagged wheel matches the measured installation; the selector still performs CPU inference. `torchvision` and `torchaudio` are not required. On Linux, use Python 3.12, activate with `source .venv/bin/activate`, then use the same pip commands. The published exact-choice checks were performed on the recorded Windows environment; no clean-environment or cross-platform installation is claimed merely from the version list.

Training/test tools are optional: `python -m pip install -r development/requirements.txt`. The external-reference SDK list is separate and unnecessary for scoring. No installation command downloads LLM weights.

## 4. Recommended score reproduction

```powershell
python -B frozen/reproduce.py --split validation --output-dir ../mp2-reproduction/validation
python -B frozen/reproduce.py --split test --output-dir ../mp2-reproduction/test
```

The standard-library controller sets numeric-library threads to one, hides GPUs, removes inherited development guard variables/PYTHONPATH and starts the unchanged official prediction and scoring scripts as separate fresh processes. It checks the release hashes before running. Stored published predictions are read only by the controller **after** new predictions and scoring finish, to verify identical results; the selection function never reads them.

Expected results:

- Validation: `correct=743`, `n=798`, `accuracy=93.10776942355889`.
- Test: `correct=374`, `n=467`, `accuracy=80.08565310492506`.
- The validation disagreement subset is 300/355; it is a development diagnostic, not the reported complete-test metric.

Choose an explicit local output directory with `--output-dir`. It may be inside or outside the repository: for example, `--output-dir review_output/test` or `--output-dir ../mp2-reproduction/test`. The script creates missing directories. New predictions, official scoring JSON, raw stdout/stderr, temporary files and the external wall-clock summary all stay in the selected directory. Existing predictions or scores are not overwritten. For an additional repeat, choose a new directory. The submitted repository does not include the author's local verification outputs.

An unchanged official data check is available as `python -B code/check_data.py`. It reads reference files for alignment checking; it is not a training step and is separate from the measured prediction-plus-scoring pair.

## 5. Direct official commands

To use the evaluator without the convenience controller, set the environment before Python starts. PowerShell:

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
$env:PYTHONHASHSEED="0"
$env:OMP_NUM_THREADS="1"
$env:OMP_THREAD_LIMIT="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:BLIS_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"
$env:VECLIB_MAXIMUM_THREADS="1"
$env:OMP_MAX_ACTIVE_LEVELS="1"
$env:OMP_DYNAMIC="FALSE"
$env:MKL_DYNAMIC="FALSE"
$env:CUDA_VISIBLE_DEVICES="-1"
$env:HIP_VISIBLE_DEVICES="-1"
python -B code/predict.py --method code/submission.py --checkpoint frozen/model_bundle --split test --output ../mp2-reproduction/direct-test-predictions.json
python -B code/score_predictions.py --split test --predictions ../mp2-reproduction/direct-test-predictions.json --output ../mp2-reproduction/direct-test-score.json
```

For Bash, export the same variables or use the provided controller. Substitute `--split validation` in both commands for validation. Report the official scorer's `accuracy` field in percent; do not infer a score from a predictions file. Timing fields in JSON can vary; protocol, split, ID order, choices, correct count and accuracy should agree.

The frozen method accepts an independent audit bank without retraining:

```text
python -B code/predict.py --method code/submission.py --checkpoint frozen/model_bundle --split audit --data-dir /path/to/audit_inputs --output ../mp2-reproduction/audit-predictions.json
```

An audit input bank needs only the official-format manifest and candidate text/embedding files. No audit reference answers are required for prediction. The selector uses the supplied bank size and does not use question IDs to retrieve answers.

## 6. Resource accounting and measured results

`frozen/results/resource_report.json` records the original full frozen evaluation, including its software guard overhead. Prediction and scoring ran in separate fresh processes on **AMD Ryzen 9 7940HX, Windows, one numeric thread**. Continuous pair wall times were **11.6254822 s** and **11.6061212 s**; process-sum times were 11.6097404 s and 11.5913890 s. Local timing passed the 30-second limit. The specified Xeon reference machine was not measured.

The nine runtime assets total **7,158,695 bytes** by summing each file's conservative stored-or-expanded size. Exact NPZ-member expansion including `.npy` headers plus the raw tree/config files is **7,135,173 bytes**. Both are below **16,777,216 bytes**. Learned transforms, normalization statistics, all ensemble members and configuration are included. Ordinary dependencies, supplied data, archived experimental models and output predictions are not loaded as additional model assets. Local package verification outputs are kept outside the submission repository and do not replace the original evaluation records.

The original software guard was an additional audit mechanism, not an OS airgap. It tested reference/network denials. The public selector has no network or reference-reading code path; the official scorer reads references independently. No GPU inference, new LLM generation or API call is part of reproduction.

## 7. Development evidence and its limits

`development/src/experiments/` contains the historical batch scripts; `development/tests/` contains the corresponding tests. The retained training lineage is B12 for attention members, B15 for R5/SVD, and B17/B20 for frozen fusion and interface validation. Batch JSON records and `seed_registry.json` preserve seeds, hyperparameters, training/search costs where recorded, outcomes and rejected alternatives. Earlier environments and partial timing records remain historical; use the final resource report for the published model's complete-process measurements.

These archived scripts are **reference material, not the score-reproduction workflow**. Some expect their original workspace paths or intentionally omitted internal specification files. Archive copies normalize private specification references and workstation paths; this is recorded with original/published hashes in `development/publication_manifest.json`. Running the whole archived test suite or historical freeze driver can start training, recreate experiments or reference unavailable internal inputs. It is unnecessary for checkpoint evaluation. The archive is not presented as a clean, one-command retraining pipeline.

Structured experiment results, candidate choices and learned checkpoints are included. Internal planning/specification documents, embedded private planning fields, miscellaneous private text, raw console logs, caches, temporary environments and duplicate fixtures are excluded. The historical source prompts used by the external judge are not supplied as executable reproduction inputs. Historical SHA fields may identify original artifacts before publication normalization; use the publication manifest for the actual archived copy hashes.

`development/src/reference/` and `development/runs/ref/` describe a separate validation-only external-LLM ceiling/reference experiment, outside the submission's inference resource constraints. They are not imported by the frozen selector and must not be used as training labels, additional inference evidence or ensemble members. No reference API run is needed to reproduce the submitted score. Early test baseline/distribution summaries were descriptive only; the frozen result is reproduced with the complete fixed official test bank, without using test labels for training, tuning or selecting an alternative.

## 8. Attribution and report status

AI assistance supported implementation, experiment planning and documentation. The author is responsible for understanding the method and reporting the results. Supplied GSM8K/Qwen attribution and the supplied data license remain in `code/data/LICENSE.txt` and `code/data/NOTICE.txt`. `Final_Report.md` is intentionally empty for now and must be completed separately before the final coursework submission.

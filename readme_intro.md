# Selecting Reliable LLM Reasoning

This project implements the DASE7506 MP2 task: select one of four supplied mathematical solutions using the candidate text and fixed 1024-dimensional representations. The official evaluator scores the selected final numeric answer.

I developed the system from majority voting and feature-based baselines through independent verification, cross-candidate attention, symbolic checks and model fusion. Experiments, ablations, unsuccessful alternatives and verification records are archived in `development/`. The selected method is the historical B17/B20 three-view ensemble: two attention views, each averaged across seeds 42/43/44, and an R5 tree view. It uses fixed fusion weights, answer-format masking and answer-cluster aggregation.

The frozen result is **374/467 on test (80.085653%)**, reproduced with exactly the same choices in a second fresh process. Validation accuracy is **743/798 (93.107769%)**; the disagreement subset is **300/355 (84.507042%)**.

The nine inference assets occupy **7,158,695 bytes (6.827 MiB)** under conservative accounting. The original full test prediction-plus-scoring runs took **11.625 s** and **11.606 s** on an AMD Ryzen 9 7940HX / Windows system using one numeric thread. These are local measurements; the specified Xeon reference machine was not available.

Start with [readme_reproduction.md](readme_reproduction.md). `code/` preserves the official tools and data; only its editable `submission.py` is replaced. `frozen/` is the runnable release. `development/` is evidence, not the evaluation entrypoint. `Final_Report.md` is intentionally empty pending the final report.

AI assistance was used for coding, experiment planning and documentation. A separate validation-only external-LLM reference experiment is archived for analysis; its outputs are not inference assets, training labels or ensemble members. Evaluation of the frozen submission is offline and generates no new LLM responses. Questions derive from GSM8K and candidate representations from Qwen3-0.6B; the supplied attribution and license files are retained in `code/data/`.

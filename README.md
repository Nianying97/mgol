# MGOL

PyTorch implementation of **Beyond a Single Global Geometry: Manifold-Guided
Overlapping Local Geometry Learning for Multi-Domain IoV Intrusion Detection**.

MGOL learns from heterogeneous domains using private input encoders and
classification heads, a shared Fisher-Rao temporal module, angular/semantic
routing over overlapping local transformations, and temporal self-attention.
Inference requires only the input and identifier of the current domain.


## Project structure

```text
.
├── main.py                      # Check configuration, train, predict
├── data_process.py              # Raw-unit splits and raw-window sequences
├── mgol.py                      # MGOL model
├── trainer.py                   # Two-stage optimization and evaluation
├── config.py                    # Private configuration validation
├── config.template.json         # Placeholders; not an experimental preset
├── config.schema.json           # Configuration field types
├── utils.py                     # Seeding, metrics, JSON output
├── requirements.txt
├── docs/
│   └── METHOD_ALIGNMENT.md       # Equations and implementation choices
└── tests/
    └── test_mgol.py              # Synthetic correctness tests
```

## Installation

Use Python 3.9 or newer and install the direct dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment before running the second command. For CUDA, install a
PyTorch build compatible with your device using the PyTorch installation tools.
The dependency ranges are compatibility requirements, not an exact experimental
environment lock file.

## Data protocol

Each domain supplies its own CSV, label column, and explicit benign label.
Labels remain independent across domains. The intended domains are CICIoV2024,
VeReMi2026, CICIoMT2024, and 5GNIDD2025; their input dimensions are inferred from
the supplied CSVs instead of being hard-coded.

1. Preserve the original chronological record order and any file, run, or
   session identifiers in the input CSV.
2. When reliable collection identifiers exist, set `group_columns` to those
   columns. The tuple of these fields identifies one indivisible collection
   unit. Set `block_size` to `null`.
3. Otherwise, set `group_columns` to `[]` and explicitly provide `block_size`.
   The program divides raw record order into contiguous non-overlapping blocks.
   It cannot recover missing provenance from an already merged CSV.
4. Split complete units into a training pool and independent test set. Extract
   validation units from the training pool. Class coverage is checked at raw
   record level and again after sequence labeling. An impossible split fails
   explicitly rather than falling back to a random-window split.
5. Fit numeric missing-value medians, categorical vocabularies, means and standard
   deviations on the final training subset only. Apply the same transformations
   to validation, test, and later inference. Unknown categories map to an
   unknown code; no vocabulary or statistic is fitted on held-out samples.
6. Construct local windows inside each unit. Flatten each window in
   record-major, field-minor order. Combine consecutive windows within the same
   unit into a sequence. The sequence label is the final raw record of its final
   window. A sequence never crosses a collection unit or split boundary.

There is **no handcrafted feature reconstruction**: no CAN histograms, Gaussian
parameters, window statistics, or VeReMi motion-derived features. Setting
`feature_columns` to `null` retains all original columns except the label and
collection identifiers. An explicit list can describe a previously prepared CSV
schema; no feature selection is fitted or silently applied. With `F` raw fields
and window length `W`, the encoder input dimension per time step is `W * F`.
The local windows may overlap inside one split; their raw records cannot overlap
across splits.

## Private configuration

Copy `config.template.json` to a location **outside the public repository**, for
example `../private/config.private.json`. Fill every required setting using your
own local configuration. Numeric `null` values in the template are placeholders
and intentionally fail validation; the template is not runnable as an experiment.
Add one domain object per dataset; domain names use letters, digits and
underscores. JSON paths may use forward slashes on Windows. Relative CSV paths
are resolved against the private configuration file's directory.

`test_fraction` is the fraction of complete collection units held out as test;
`validation_fraction_of_train` is the fraction of the remaining training units
used for validation. Unequal collection sizes mean record-level ratios are
approximate. `window_size`, `stride`, and model `seq_len` jointly define the raw
span of a sequence. A fallback block must be at least that long and contain
enough independent units of each class for all three splits.

All architectural widths, units, temperatures' projection dimension, dropout,
batch sizes, epoch counts and optimization settings are explicit configuration
fields. The shared fine-tuning learning rate is derived as one tenth of the
pretraining rate, as required by the method. Temperatures and temporal
enhancement strength are learned by the model.

```bash
python main.py check-config --config ../private/config.private.json
python main.py train --config ../private/config.private.json --output-dir ../private/run_001 --device cuda:0
```

Use `--device cpu` for CPU training. Output directories must be new or empty.
Training drops incomplete final mini-batches to avoid singleton BatchNorm
updates; validation and test retain all samples. A training split smaller than
one batch is rejected.

Stage 1 alternates domains with one cross-entropy update per domain. Stage 2
samples a batch from every active domain and averages the per-domain
cross-entropy losses before one backward pass. Each epoch uses the smallest
domain loader's number of batches, giving every domain the same number of
updates. There are no alignment, entropy, confidence, smoothness or routing
regularization losses. Each stage runs its configured epoch budget, selects the
best mean-domain validation macro-F1 checkpoint, and restores it. The test set
is evaluated only after the final stage.

## Outputs and independent-domain inference

Private run directories contain `pretrain_best.pt`, `finetune_best.pt`,
`history.json`, and `test_metrics.json`. Checkpoints include fitted preprocessing
and model settings so that later inference does not refit anything. Keep these
files private along with the experiment configuration.

Metrics are accuracy, macro-F1 over the complete domain label set, and security
FPR: benign records predicted as any attack divided by all benign records.
Fractions are reported on the `[0, 1]` scale. A missing benign denominator gives
JSON `null`, not a fabricated zero.

```bash
python main.py predict --checkpoint ../private/run_001/finetune_best.pt --domain domain_1 --csv ../private/inference.csv --output ../private/predictions.json --batch-size 16 --device cpu
```

The batch size above is an inference usage example, not an experimental setting.
The input can omit the label column but must retain the trained raw fields and
collection identifiers. Predictions follow collection-unit first appearance,
then chronological sequence order. Each prediction includes its label and
probabilities in the recorded class order. Only the selected domain branch is
used; other domains need not supply inputs.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests use deliberately small synthetic configurations. They check the geometric
equations, finite gradients, raw-unit isolation, final-window labeling,
train-only transformations, raw-window flattening, domain-private gradients,
joint versus independent inference, two-stage training, checkpoint restoration,
and FPR semantics. Synthetic test scores are not manuscript results.

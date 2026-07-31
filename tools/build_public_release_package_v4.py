#!/usr/bin/env python3
"""Build a privacy-screened public code release for the PK-SSM manuscript.

The builder uses an explicit allowlist. It never copies raw wearable records,
participant/session manifests, model checkpoints, caches, or per-origin forecasts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "release_package_v4"
PACKAGE_NAME = "pk_ssm_heart_rate_forecasting_v1.0.0"
PACKAGE_DIR = RELEASE_ROOT / PACKAGE_NAME
ZIP_PATH = RELEASE_ROOT / f"{PACKAGE_NAME}.zip"

TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".gitignore",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".joblib",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
}
IDENTIFIER_FIELD = re.compile(
    r"^(participant|session|user|subject|origin|workout|athlete|record)(_|-)?id$",
    re.IGNORECASE,
)
ABSOLUTE_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/])")
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret|password)"
    r"\s*[:=]\s*['\"][^'\"]{8,}['\"]"
)
MAX_PUBLIC_FILE_BYTES = 20 * 1024 * 1024


ROOT_FILES = [
    "ACTIVITY_SHIFT_RESULTS_V4.md",
    "CLAIM_EVIDENCE_MAP_V4.md",
    "FIGURE_LEGENDS_V4.md",
    "TERMINOLOGY_LEDGER_V4.md",
]

PROTOCOL_FILES = [
    "GOLDENCHEETAH_LIGHTWEIGHT_LONGITUDINAL_PROTOCOL_V1.md",
    "PROTOCOL_AMENDMENT_008_GOLDENCHEETAH_PERSONALIZATION_GATE.md",
    "PROTOCOL_AMENDMENT_009_VALIDATION_BASELINE_BOUNDARY.md",
    "PROTOCOL_AMENDMENT_010_FIVE_FOLD_UNSEEN_USER_VALIDATION.md",
    "PROTOCOL_AMENDMENT_011_COMPLETE_V4_BASELINE_VALIDATION.md",
    "PROTOCOL_AMENDMENT_012_PKSSM_ARCHITECTURE_SCREEN_LOCK.md",
    "PROTOCOL_AMENDMENT_013_PKSSM_TCN_SEED_STABILITY_LOCK.md",
    "PROTOCOL_AMENDMENT_014_CALIBRATION_OPENING_AND_TEST_SPECIFICATION.md",
    "PROTOCOL_AMENDMENT_015_LOCKED_EVALUATION_CODE_HASH.md",
    "PROTOCOL_AMENDMENT_016_SEALED_TEST_OPENING_DECISION.md",
    "PROTOCOL_AMENDMENT_017_TEST_SUMMARY_CODE_LOCK.md",
    "PROTOCOL_AMENDMENT_018_RIDGE_TEST_EVALUATOR_LOCK.md",
    "PROTOCOL_AMENDMENT_019_SEALED_TEST_RESULTS_LOCK.md",
    "PROTOCOL_AMENDMENT_020_GOLDENCHEETAH_EXTERNAL_TEST_OPENING.md",
    "PROTOCOL_AMENDMENT_021_GOLDENCHEETAH_SUMMARY_CODE_LOCK.md",
    "PROTOCOL_AMENDMENT_022_GOLDENCHEETAH_SPORT_SUMMARY_LOCK.md",
    "PROTOCOL_AMENDMENT_024_KINETIC_PARAMETER_AUDIT_LOCK.md",
    "PROTOCOL_AMENDMENT_025_KINETIC_PARAMETER_RESULTS_LOCK.md",
    "PROTOCOL_AMENDMENT_026_WEARABLE_ACTIVITY_SHIFT_LOCK.md",
    "PROTOCOL_AMENDMENT_027_ACTIVITY_SHIFT_CACHE_CORRECTION.md",
]

SCRIPT_FILES = [
    "audit_goldencheetah_longitudinal_eligibility.py",
    "audit_locked_kinetic_parameters.py",
    "audit_wearable_exercise.py",
    "build_figure6_v4.py",
    "build_figures_v4.py",
    "build_goldencheetah_transition_cache_v1.py",
    "build_primary_transition_cache_v4.py",
    "build_wearable_activity_shift_cache_v1.py",
    "build_wearable_activity_shift_cache_v2.py",
    "evaluate_locked_ridge_test.py",
    "export_activity_shift_forecasts_v1.py",
    "export_locked_forecasts.py",
    "freeze_goldencheetah_light_subset_v1.py",
    "freeze_wearable_exercise_splits.py",
    "freeze_wearable_exercise_splits_v4.py",
    "run_activity_shift_exports_v2.py",
    "run_activity_shift_model_selection_v2.py",
    "run_pretest_model_selection.py",
    "run_ridge_summary_baseline.py",
    "score_calibrated_forecasts.py",
    "summarize_activity_shift_results_v1.py",
    "summarize_activity_shift_results_v2.py",
    "summarize_goldencheetah_sport_groups.py",
    "summarize_locked_goldencheetah_results.py",
    "summarize_locked_v4_results.py",
]

RESULT_FILES = {
    "outputs/locked_summary_v1": [
        "paired_participant_inference.json",
        "primary_ensemble_summary.csv",
        "primary_seed_differences.csv",
        "primary_seed_summary.csv",
        "secondary_fixed_seed_summary.csv",
        "subgroup_fold_summary.csv",
        "summary.json",
        "uncertainty_summary.csv",
    ],
    "outputs/locked_kinetic_parameter_audit_v1": [
        "parameter_correlations.csv",
        "parameter_summary.csv",
        "summary.json",
    ],
    "outputs/goldencheetah_locked_summary_v1": [
        "model_summary.csv",
        "paired_inference.json",
        "subgroup_summary.csv",
        "summary.json",
        "uncertainty_summary.csv",
    ],
    "outputs/goldencheetah_sport_summary_v1": [
        "sport_model_summary.csv",
        "sport_paired_inference.json",
        "summary.json",
    ],
    "outputs/activity_shift_locked_summary_v2_final": [
        "activity_shift_summary.json",
    ],
}

SPLIT_AUDIT_FILES = [
    "README.md",
    "fold_balance_audit.csv",
    "leakage_audit.json",
    "split_policy.json",
]


README = """# PK-SSM heart-rate transition forecasting

This repository accompanies the manuscript **Deployment-Aware Evaluation of Physiology-Guided Heart-Rate Transition Forecasting under User and Activity Shifts**, prepared for *Biomedical Signal Processing and Control*.

Public repository: https://github.com/pz565893-ui/pk-ssm-heart-rate-forecasting

## Scope

The repository contains the PK-SSM implementation, comparator models, the locked
candidate configuration and protocol records, split-generation and leakage-audit code,
aggregate evaluation outputs, and de-identified source data for Figures 1-6.

The release intentionally does **not** contain raw wearable records,
participant- or session-level manifests, cached tensors, model checkpoints,
per-origin predictions, or local machine paths. Raw data must be obtained from
the original providers under their respective terms.

## Data sources

1. Wearable Device Dataset from Induced Stress and Structured Exercise Sessions,
   PhysioNet, version 1.0.1, DOI: https://doi.org/10.13026/he0v-tf17
2. GoldenCheetah OpenData, OSF, DOI: https://doi.org/10.17605/OSF.IO/6HFPZ

See `DATA_ACCESS.md` for the data boundary and expected local preparation.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the virtual environment using the command appropriate for the local
operating system before running the pipeline.

## Repository map

- `transition_forecasting/`: PK-SSM, baselines, datasets, training, conformal
  calibration, and transition metrics.
- `scripts/`: split, cache, model-selection, evaluation, summary, and figure code.
- `configs/`: locked PK-SSM candidate grid.
- `protocol/`: frozen experiment policy and chronological amendments.
- `outputs/`: aggregate, non-identifying locked results only.
- `splits/wearable_exercise_v4/`: public split policy and aggregate audits only.
- `figures_v4/`: publication figures and de-identified source data.

## Reproduction entry points

Each command exposes its accepted arguments through `--help`. Run commands from
the repository root.

```bash
python scripts/freeze_wearable_exercise_splits_v4.py --help
python scripts/freeze_goldencheetah_light_subset_v1.py --help
python scripts/build_primary_transition_cache_v4.py --help
python scripts/build_goldencheetah_transition_cache_v1.py --help
python scripts/run_pretest_model_selection.py --help
python scripts/run_activity_shift_model_selection_v2.py --help
python scripts/summarize_locked_v4_results.py --help
python scripts/summarize_locked_goldencheetah_results.py --help
python scripts/build_figures_v4.py --help
python scripts/build_figure6_v4.py --help
```

The complete order, input boundaries, and output mapping are documented in
`REPRODUCIBILITY.md`. Full model refitting requires locally downloaded source
data and substantially more compute than reusing the included aggregate outputs.

## Privacy and provenance

`PACKAGE_AUDIT.json`, `MANIFEST.csv`, and `SHA256SUMS.txt` document the public
allowlist, privacy checks, file sizes, and cryptographic hashes used for this
release. The release builder is included at
`tools/build_public_release_package_v4.py`.

## Citation

Use `CITATION.cff` for author and manuscript metadata. Add the Zenodo DOI after
archiving the release.

## License

The software in this repository is released under the MIT License. Dataset terms
remain governed by the original data providers. See `LICENSE`.
"""


DATA_ACCESS = """# Data access and redistribution boundary

## Primary dataset

Wearable Device Dataset from Induced Stress and Structured Exercise Sessions,
PhysioNet version 1.0.1: https://doi.org/10.13026/he0v-tf17

## External dataset

GoldenCheetah OpenData, OSF: https://doi.org/10.17605/OSF.IO/6HFPZ

Download both datasets directly from their providers and follow their current
terms of use. This repository does not redistribute raw records or reconstructed
individual trajectories.

The public release contains split-generation code, the frozen split policy,
aggregate leakage audits, and de-identified figure source data. Participant- and
session-level role manifests remain private because pseudonymous identifiers are
still individual-level metadata.

Store downloaded records outside version control. The public `.gitignore`
excludes common raw-data, cache, checkpoint, and prediction directories.
"""


REPRODUCIBILITY = """# Reproducibility workflow

Run the workflow from the repository root after obtaining the source datasets.
Use each script's `--help` output for exact local paths and resource options.

1. Audit the downloaded datasets with `audit_wearable_exercise.py` and
   `audit_goldencheetah_longitudinal_eligibility.py`.
2. Generate the primary leakage-controlled roles with
   `freeze_wearable_exercise_splits_v4.py`.
3. Freeze the lightweight longitudinal external subset with
   `freeze_goldencheetah_light_subset_v1.py`.
4. Build transition caches with `build_primary_transition_cache_v4.py`,
   `build_goldencheetah_transition_cache_v1.py`, and the activity-shift cache
   scripts.
5. Run source-only model selection with `run_pretest_model_selection.py` and
   `run_activity_shift_model_selection_v2.py`.
6. Export locked forecasts and apply source-only conformal calibration with the
   export and scoring scripts.
7. Produce aggregate tables with the three `summarize_*` workflows.
8. Generate Figures 1-5 with `build_figures_v4.py` and Figure 6 with
   `build_figure6_v4.py`.

The chronological protocol and amendment files in `protocol/` define the locked
decision order. Public aggregate outputs can be checked against `MANIFEST.csv`
and `SHA256SUMS.txt`. Raw records, individual role manifests, cached tensors,
checkpoints, and per-origin forecasts are deliberately excluded.
"""


UPLOAD_GUIDE = """# GitHub and Zenodo upload guide

## Before uploading

1. Open `PACKAGE_AUDIT.json` and confirm that `status` is `passed`.
2. Confirm that the included `LICENSE` file identifies the MIT License.
3. Upload the **contents** of this folder as the repository root. Do not upload
   the outer release-package directory or any original project directory.

## GitHub

1. Create a new public repository with no auto-generated README, license, or
   `.gitignore` because those files are already staged here.
2. Upload this package's contents, or use Git locally, and commit them to `main`.
3. Confirm that no raw-data, cache, checkpoint, or private-manifest directory is
   visible on GitHub.
4. Copy the final repository URL into the manuscript and `CITATION.cff`.

## Zenodo archival release

1. Sign in to Zenodo and enable the GitHub integration for the repository before
   publishing the first GitHub release.
2. On GitHub, create release tag `v1.0.0`, use a descriptive release title, and
   attach the provided ZIP as an optional convenience artifact.
3. Allow Zenodo to archive the release, then complete the metadata using the two
   authors and the manuscript title in `CITATION.cff`.
4. Add the resulting Zenodo DOI to the manuscript, cover letter, repository
   README, and `CITATION.cff`.
5. Keep the version-specific DOI for the submitted code release; the concept DOI
   may also be retained for later versions.

Do not submit the manuscript until the repository URL and archived DOI replace
all remaining repository placeholders in the submission files.
"""


CITATION_CFF = """cff-version: 1.2.0
message: "If you use this software, please cite the associated manuscript."
title: "PK-SSM heart-rate transition forecasting"
type: software
version: 1.0.0
repository-code: "https://github.com/pz565893-ui/pk-ssm-heart-rate-forecasting"
license: MIT
authors:
  - family-names: "Pang"
    given-names: "Keren"
    orcid: "https://orcid.org/0009-0007-2506-9206"
    email: "20248657@o.shinhan.ac.kr"
    affiliation: "Department of Sports & Health Science, Shinhan University"
  - family-names: "Min"
    given-names: "Changrong"
    email: "mcr19940816@gmail.com"
    affiliation: "Criminal Investigation Police University of China"
preferred-citation:
  type: article
  title: "Deployment-Aware Evaluation of Physiology-Guided Heart-Rate Transition Forecasting under User and Activity Shifts"
  authors:
    - family-names: "Pang"
      given-names: "Keren"
      orcid: "https://orcid.org/0009-0007-2506-9206"
    - family-names: "Min"
      given-names: "Changrong"
  journal: "Biomedical Signal Processing and Control"
"""


PYPROJECT = """[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pk-ssm-heart-rate-forecasting"
version = "1.0.0"
description = "Personalized kinetic state-space heart-rate transition forecasting"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
  "numpy>=2.0",
  "pandas>=2.2",
  "scipy>=1.14",
  "scikit-learn>=1.5",
  "xgboost>=2.1",
  "matplotlib>=3.9",
  "torch>=2.6",
]

[tool.setuptools.packages.find]
include = ["transition_forecasting*", "scripts*"]
"""


PUBLIC_GITIGNORE = """# Raw or restricted data
data/
raw_data/
downloads/
private_manifests/
*participant_manifest*.csv
*session_manifest*.csv
*history_manifest*.csv
*role_manifest*.csv

# Derived large or individual-level artifacts
cache/
caches/
checkpoints/
predictions/
forecasts/
*.npy
*.npz
*.parquet
*.pkl
*.pickle
*.joblib
*.pt
*.pth
*.ckpt

# Python and local environments
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.mypy_cache/

# Editors and operating systems
.idea/
.vscode/
.DS_Store
Thumbs.db
"""


MIT_LICENSE = """MIT License

Copyright (c) 2026 PANG KEREN and MIN CHANGRONG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def require_file(relative: str) -> Path:
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(f"Required public-release file is missing: {relative}")
    return source


def normalized_text(source: Path) -> str:
    text = source.read_text(encoding="utf-8-sig")
    root_windows = str(ROOT)
    root_posix = ROOT.as_posix()
    text = text.replace(root_windows, ".")
    text = text.replace(root_windows.lower(), ".")
    text = text.replace(root_posix, ".")
    return text


def copy_file(relative: str, destination_relative: str | None = None) -> None:
    source = require_file(relative)
    target = PACKAGE_DIR / (destination_relative or relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix in TEXT_SUFFIXES or source.name == ".gitignore":
        target.write_text(normalized_text(source), encoding="utf-8", newline="\n")
    else:
        shutil.copy2(source, target)


def write_text(relative: str, content: str) -> None:
    target = PACKAGE_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.strip() + "\n", encoding="utf-8", newline="\n")


def iter_json_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from iter_json_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_json_keys(child)


def audit_package() -> tuple[list[dict[str, Any]], list[str]]:
    manifest: list[dict[str, Any]] = []
    issues: list[str] = []
    for path in sorted(p for p in PACKAGE_DIR.rglob("*") if p.is_file()):
        relative = path.relative_to(PACKAGE_DIR).as_posix()
        suffix = path.suffix.lower()
        size = path.stat().st_size
        if suffix in FORBIDDEN_SUFFIXES:
            issues.append(f"forbidden file type: {relative}")
        if size > MAX_PUBLIC_FILE_BYTES:
            issues.append(f"file exceeds 20 MiB: {relative}")

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        category = relative.split("/", 1)[0]
        manifest.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": digest,
                "category": category,
            }
        )

        is_text = suffix in TEXT_SUFFIXES or path.name == ".gitignore"
        if not is_text:
            continue
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            issues.append(f"declared text file is not UTF-8: {relative}")
            continue
        if ABSOLUTE_WINDOWS_PATH.search(content):
            issues.append(f"absolute Windows path found: {relative}")
        profile_forward = "C:" + "/Users/"
        profile_backward = "C:" + "\\Users\\"
        local_account = "\u5e9e\u76f4\u6e9c"
        if profile_forward in content or profile_backward in content:
            issues.append(f"user-profile path found: {relative}")
        if local_account in content:
            issues.append(f"local Windows account name found: {relative}")
        if SECRET_ASSIGNMENT.search(content):
            issues.append(f"possible embedded credential found: {relative}")

        if suffix == ".csv":
            rows = csv.reader(content.splitlines())
            header = next(rows, [])
            restricted = [field for field in header if IDENTIFIER_FIELD.match(field.strip())]
            if restricted:
                issues.append(
                    f"individual identifier column(s) {restricted!r} found: {relative}"
                )
        elif suffix == ".json":
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                issues.append(f"invalid JSON in {relative}: {exc}")
            else:
                restricted = sorted(
                    {key for key in iter_json_keys(payload) if IDENTIFIER_FIELD.match(key)}
                )
                if restricted:
                    issues.append(
                        f"individual identifier JSON key(s) {restricted!r} found: {relative}"
                    )
    return manifest, issues


def write_manifest(manifest: list[dict[str, Any]]) -> None:
    manifest_path = PACKAGE_DIR / "MANIFEST.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256", "category"])
        writer.writeheader()
        writer.writerows(manifest)

    checksums = "\n".join(f"{row['sha256']}  {row['path']}" for row in manifest) + "\n"
    (PACKAGE_DIR / "SHA256SUMS.txt").write_text(checksums, encoding="utf-8", newline="\n")


def main() -> int:
    if PACKAGE_DIR.exists():
        prior_audit_path = PACKAGE_DIR / "PACKAGE_AUDIT.json"
        if not prior_audit_path.is_file():
            raise FileExistsError(
                f"Release target already exists without a failed audit: {PACKAGE_DIR}"
            )
        prior_audit = json.loads(prior_audit_path.read_text(encoding="utf-8"))
        if prior_audit.get("package") != PACKAGE_NAME or prior_audit.get("status") != "failed":
            raise FileExistsError(
                f"Release target is not a verified failed staging package: {PACKAGE_DIR}"
            )
        if PACKAGE_DIR.parent.resolve() != RELEASE_ROOT.resolve() or PACKAGE_DIR.name != PACKAGE_NAME:
            raise RuntimeError("Refusing to clean an unexpected staging path")
        shutil.rmtree(PACKAGE_DIR)
    if ZIP_PATH.exists():
        raise FileExistsError(f"Release ZIP already exists: {ZIP_PATH}")
    PACKAGE_DIR.mkdir(parents=True)

    for relative in ROOT_FILES:
        copy_file(relative, f"docs/{relative}")
    for relative in PROTOCOL_FILES:
        copy_file(relative, f"protocol/{relative}")

    for module in sorted((ROOT / "transition_forecasting").glob("*.py")):
        copy_file(module.relative_to(ROOT).as_posix())
    for script in SCRIPT_FILES:
        copy_file(f"scripts/{script}")
    copy_file("scripts/build_public_release_package_v4.py", "tools/build_public_release_package_v4.py")
    write_text("scripts/__init__.py", '"""Command-line workflows for the PK-SSM release."""')

    for config in sorted((ROOT / "configs").glob("*.json")):
        copy_file(config.relative_to(ROOT).as_posix())

    for directory, filenames in RESULT_FILES.items():
        for filename in filenames:
            copy_file(f"{directory}/{filename}")
    for filename in SPLIT_AUDIT_FILES:
        copy_file(f"splits/wearable_exercise_v4/{filename}")

    figures = ROOT / "figures_v4"
    for figure in sorted(figures.glob("Figure*")):
        if figure.is_file() and figure.suffix.lower() in {".pdf", ".png", ".svg"}:
            copy_file(figure.relative_to(ROOT).as_posix())
    for source_data in sorted((figures / "source_data").glob("*")):
        if source_data.is_file() and source_data.suffix.lower() in {".csv", ".txt"}:
            copy_file(source_data.relative_to(ROOT).as_posix())

    copy_file("experiment_requirements.txt", "requirements.txt")
    write_text("README.md", README)
    write_text("DATA_ACCESS.md", DATA_ACCESS)
    write_text("REPRODUCIBILITY.md", REPRODUCIBILITY)
    write_text("UPLOAD_GITHUB_ZENODO.md", UPLOAD_GUIDE)
    write_text("CITATION.cff", CITATION_CFF)
    write_text("pyproject.toml", PYPROJECT)
    write_text(".gitignore", PUBLIC_GITIGNORE)
    write_text("LICENSE", MIT_LICENSE)

    manifest, issues = audit_package()
    audit = {
        "package": PACKAGE_NAME,
        "built_on": date.today().isoformat(),
        "status": "failed" if issues else "passed",
        "file_count_before_manifest": len(manifest),
        "total_bytes_before_manifest": sum(row["bytes"] for row in manifest),
        "checks": [
            "explicit allowlist only",
            "no forbidden cache/checkpoint formats",
            "no file above 20 MiB",
            "no absolute Windows or user-profile paths in text files",
            "no obvious embedded credentials",
            "no participant/session/user/subject/origin/workout identifier fields in CSV or JSON",
        ],
        "deliberately_excluded": [
            "raw wearable records",
            "participant- and session-level split manifests",
            "cached tensors and arrays",
            "model checkpoints",
            "per-origin and per-participant predictions",
            "local environment files and machine paths",
        ],
        "issues": issues,
    }
    write_text("PACKAGE_AUDIT.json", json.dumps(audit, indent=2, ensure_ascii=True))
    if issues:
        print(json.dumps(audit, indent=2, ensure_ascii=True))
        return 2

    manifest, post_audit_issues = audit_package()
    if post_audit_issues:
        print(json.dumps({"status": "failed", "issues": post_audit_issues}, indent=2))
        return 2
    write_manifest(manifest)

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in PACKAGE_DIR.rglob("*") if p.is_file()):
            archive.write(path, f"{PACKAGE_NAME}/{path.relative_to(PACKAGE_DIR).as_posix()}")

    summary = {
        "status": "passed",
        "package_directory": str(PACKAGE_DIR),
        "zip_file": str(ZIP_PATH),
        "public_files": len([p for p in PACKAGE_DIR.rglob("*") if p.is_file()]),
        "zip_bytes": ZIP_PATH.stat().st_size,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

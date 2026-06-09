# External Databases

This project is trained and validated against three **public** PPG datasets. The raw
data files are **not redistributed in this repository** (each carries its own licence and
some are large); download them from the official sources below. The helper scripts
`backend/download_bidmc.py` and `backend/download_ppg_dalia.py` fetch two of them
automatically into a local `external_data/` folder that is git-ignored.

The self-collected Vietnamese cohort recordings used for per-subject BP calibration are
**not shared** — they contain personally identifiable information (real participant names)
and are excluded from version control.

## 1. PPG-BP Database (Liang et al., 2018) — BP training data

219 subjects / 657 recordings, fingertip transmission PPG @ 1 kHz, with reference cuff
systolic/diastolic blood pressure. Used to train the Random-Forest / SVR blood-pressure
regressor (`ml/train_ppg_bp.py`).

- **Dataset (figshare):** https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299
- **DOI:** https://doi.org/10.6084/m9.figshare.5459299
- **Paper (Scientific Data 2018):** https://www.nature.com/articles/sdata201820
- **Licence:** CC BY 4.0

## 2. BIDMC PPG and Respiration Dataset — cross-dataset SQA validation

53 ICU recordings (PhysioNet), used as an out-of-distribution test set for the
signal-quality classifier (leave-one-dataset-out evaluation).

- **Source:** https://physionet.org/content/bidmc/1.0.0/
- **Download helper:** `python backend/download_bidmc.py`
- **Licence:** PhysioNet (open, free for research)

## 3. PPG-DaLiA — cross-dataset SQA validation (motion / daily activity)

15 subjects performing daily activities, wrist reflectance PPG. Used as the hardest
out-of-distribution SQA test set (cadence-locked motion artefact).

- **Source:** https://archive.ics.uci.edu/dataset/495/ppg+dalia
- **Direct zip:** https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip
- **Download helper:** `python backend/download_ppg_dalia.py`
- **Licence:** CC BY 4.0 (UCI Machine Learning Repository)

## Convenience mirror (Google Drive)

A convenience copy of the public datasets used in this project is mirrored here for
reviewers. Always prefer the official sources above as the authoritative origin; this
mirror is provided only to reproduce the exact files used.

- **Google Drive folder:** https://drive.google.com/drive/folders/1Gi-SWk4dn0YXTiEr-pAsSNV7Ko50Oodz

## Local layout expected by the scripts

```
external_data/            # git-ignored
├── ppg_bp/               # PPG-BP Database.zip extracted here
├── bidmc/                # download_bidmc.py output
└── ppg_dalia/            # download_ppg_dalia.py output
```

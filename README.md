# Movie Recommender

Hybrid movie recommender with a Gradio UI, local embeddings, and optional zero-shot reranking.

## Overview

This app combines:
- semantic retrieval using **BGE small embeddings** + **ChromaDB**
- optional **zero-shot relevance scoring** using **BART large MNLI**
- weighted final ranking with filters and configurable scoring controls

All models run locally and offline.

Base idea attribution:
- Useful Video: `https://youtu.be/Q7mS1VHm3Yw?si=O4fFO3kcBROvb-XZ`

## Current Behavior

- Overviews are sanitized to avoid empty/`NaN` values in UI and metadata.
- Chroma metadata is stored with atomic types only (no list-valued fields).
- Existing ChromaDB is auto-rebuilt when old/invalid metadata is detected.
- Zero-shot relevance sorting is shown only when zero-shot scoring is enabled.
- App binds to localhost (`127.0.0.1:7860`).

## Features

- Free-text recommendation query ("describe what you want").
- Adjustable scoring weights:
  - `alpha`: semantic similarity
  - `beta`: zero-shot relevance
  - `gamma`: rating
  - `delta`: popularity
- Filters for:
  - minimum rating
  - release year range
  - one or more genres
- Optional toggles:
  - zero-shot scoring
  - popularity normalization (log scale)
  - rating normalization
  - MMR diversity (currently placeholder/experimental)
- Similar-movie lookup by title.

## Project Structure

```text
.
├── recommender.py              # Main app (DB build/load + recommendation logic + Gradio UI)
├── clean_data.ipynb            # Data preparation notebook
├── data/
│   ├── movies.csv
│   └── cleaned.csv             # Main input used by the app
├── models/
│   ├── bart-large-mnli/        # Local zero-shot model files
│   └── bge-small/              # Local embedding snapshot files
└── chroma_movies_db/           # Persisted local Chroma database (generated/updated at runtime)
```

## Requirements

- Python 3.10+ (3.11 recommended)
- `data/cleaned.csv` dataset file
- local model files:
  - `models/bart-large-mnli`
  - `models/bge-small/models--BAAI--bge-small-en-v1.5/snapshots/...`

Install dependencies:

```bash
pip install numpy pandas gradio transformers torch langchain-chroma langchain-huggingface langchain-core
```

## Quick Start

```bash
python recommender.py
```

Open:
- `http://127.0.0.1:7860`

## UI Guide

- **Describe what you want** tab:
  - enter a natural-language request (example: `a dark sci-fi thriller with strong world-building`)
  - adjust weights/filters in **Advanced Options**
  - click **Get Recommendations**
- **Find similar movies** tab:
  - enter a movie title
  - click **Find Similar**

## Data Requirements

`data/cleaned.csv` should include the columns used by the app:

- `id`
- `imdb_id`
- `title` or `original_title`
- `overview`
- `genres`
- `director`
- `cast`
- `keywords`
- `tagline`
- `vote_average`
- `release_year`
- `popularity`

## Runtime Notes

- Offline mode is enabled in code:
  - `TRANSFORMERS_OFFLINE=1`
  - `HF_HUB_OFFLINE=1`
- If `chroma_movies_db/` is missing, the app builds it from `data/cleaned.csv`.
- If old DB metadata is incompatible, the app automatically rebuilds the DB.
- MMR toggle is currently experimental (placeholder logic in current version).

## Troubleshooting

- **Missing embedding snapshot**: verify `models/bge-small/models--BAAI--bge-small-en-v1.5/snapshots/` contains at least one snapshot folder.
- **Port already in use**: change `server_port` in `recommender.py`.
- **Cold start is slow**: first run loads models and may rebuild the vector DB.

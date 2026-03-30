# Offline NUST Admissions Chatbot (Retrieval-Only)

This chatbot is fully offline and only answers from the official FAQ JSON file.

## What it does

- Uses FAQ JSON as the only source of truth.
- Intelligent conversational layer for greetings and small talk
- Hybrid retrieval for similar/paraphrased questions
- Splits compound queries into parts.
- Uses hybrid retrieval:
  - Semantic: SentenceTransformer (`all-MiniLM-L6-v2`) + FAISS
  - Keyword: BM25 (`rank-bm25`)
  - Fuzzy matching: RapidFuzz for near-duplicates
- Fuses scores with deterministic weights:
  - `final_score = 0.6 * semantic + 0.4 * keyword + margin-based filtering`
- Applies confidence filtering.
- Returns template-based responses only.
- Uses graceful "not found" fallback when confidence is low.
- Includes NUST admissions office contact on not-found replies.

## Files

- `data_loader.py` - JSON loading and validation
- `intent.py` - Conversational intent detection and templated responses
- `processor.py` - normalization and compound query splitting
- `retriever.py` - semantic + keyword retrieval and score fusion
- `responder.py` - strict template-based response generation
- `main.py` - chatbot wiring and CLI loop
- `app.py` - Professional Streamlit UI with NUST branding
- `smoke_test.py` - quick local sanity check

## Input Data

Expected FAQ file path:

- `data/nust_faq.json`

Format:

```json
[
  { "question": "...", "answer": "..." }
]
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run CLI

```bash
python main.py
```

## Run Streamlit Live Demo

```bash
streamlit run app.py
```

This opens a chat-style UI in your browser for live demos.

Logo path used by the UI:

- `assets/nust_logo.png`

Example:

```text
Ask: Is there any age limit for Undergraduate Admission?
```

## Run Smoke Test

```bash
python smoke_test.py
```

## Run Benchmark

```bash
python benchmark.py
```

## Runtime Notes

- No internet/API calls are made during runtime.
- The chatbot enforces `local_files_only=True`; it will fail fast if `all-MiniLM-L6-v2` is not present locally.
- If model files are already cached locally, everything runs offline.


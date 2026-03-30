# NUST Admissions Assistant (LLM-First)

This implementation uses a local GGUF LLM as the **primary** answer engine and grounds every answer on `nust_faq.json`.

## How it works

1. Loads FAQ knowledge base from one of:
   - `LLM/data/nust_faq.json`
   - `../data/nust_faq.json`
   - `../approach2/data/nust_faq.json`
2. Retrieves top FAQ candidates with semantic search (`sentence-transformers + FAISS`)
3. Adds typo tolerance using query normalization and fuzzy token correction
4. Sends retrieved FAQ context to local LLM (`llama-cpp-python`) to generate grounded answer
5. Falls back safely if confidence is too low or LLM fails

## Run

```bash
cd LLM
pip install -r requirements.txt
streamlit run app.py
```

## Model path

By default it searches for:
- `phi-3-mini-4k-instruct-q4_K_M.gguf`
- `Phi-3-mini-4k-instruct-q4.gguf`

in both `LLM/` and `../approach2/`.

You can also set:

```bash
export LLM_MODEL_PATH="/absolute/path/to/your-model.gguf"
```

## Notes

- This app is offline-first (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
- For typo-heavy queries, retrieval uses both semantic and fuzzy evidence before LLM generation.

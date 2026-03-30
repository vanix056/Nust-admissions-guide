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

## Performance modes

- Fast (recommended): skips LLM generation and returns grounded FAQ answers quickly.
- Accurate (LLM, slower): uses local LLM generation for nuanced responses, but may be much slower on CPU.

Choose mode from the app UI using the Response mode selector.

## Speed up LLM mode

Use these options to improve LLM speed on macOS:

1. Enable GPU offload (Metal build of llama-cpp):

```bash
export LLM_N_GPU_LAYERS=-1
```

2. Keep Fast mode for high-volume traffic, switch to Accurate mode only when needed.

3. Keep model and app on local SSD and avoid running extra heavy apps in parallel.

Notes:
- In this project, LLM speed is already tuned with smaller context and token limits.
- If `LLM_N_GPU_LAYERS=-1` is unsupported in your build, set it to `0`.

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

## Offline answers for link-based FAQs

To avoid replying with raw links when users are offline, this app uses local link knowledge:

- Source file: `data/link_offline_knowledge.json`
- During FAQ load, answers containing links are replaced with local offline summaries.

This means many link-only answers are converted to direct text responses without internet.

### Refresh workflow

1. Update `data/link_offline_knowledge.json` with new extracted page summaries.
2. Restart Streamlit app.

Optional helper script available:

```bash
cd LLM
/opt/anaconda3/bin/python build_offline_link_answers.py
```

Note: some official sites may block direct scraping (HTTP 403). In that case, keep `link_offline_knowledge.json` as the authoritative offline store.

# MTG Card Search

Semantic search over Magic: The Gathering cards and their official rulings. Describe a card in
plain English ("tap to add green mana elf") and get the closest matching cards, plus rulings that
answer rules questions ("can I respond to Grim Lavamancer's exile cost?"). Sidebar filters narrow
results by color, converted mana cost, and format legality.

## How it works

1. **Chunking** (`chunk_cards.py`): every card becomes one chunk (name, mana cost, colors, type
   line, rules text, power/toughness or loyalty), and every official ruling becomes its own chunk
   that starts with the card's name. The text that gets embedded puts the rules text first and
   spells out mana symbols (`{T}: Add {G}.` becomes "tap: Add 1 green mana."). Converted mana cost,
   colors, and format legality are stored as metadata for filtering.
2. **Subset**: the 5,000 most-played cards by EDHREC rank, with their 12,463 rulings. Change
   `MAX_CARDS` in `generate_model.py` to use more (or `None` for all 20,478).
3. **Embeddings** (`generate_model.py`): `sentence-transformers/all-MiniLM-L6-v2`, saved to `data/`.
4. **Vector database and app** (`app.py`): at startup, Streamlit loads `data/` into an in-memory
   ChromaDB collection, then embeds each query with ChromaDB's built-in ONNX copy of the same
   model, so the app doesn't need PyTorch.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | The Streamlit app |
| `requirements.txt` | Packages the app needs |
| `data/embeddings.npy`, `data/chunks.jsonl.gz` | Saved embeddings and chunks the app loads |
| `chunk_cards.py` | Turns `AllCards.json` into chunks |
| `generate_model.py` | Builds `data/` (and runs a Gradio demo) |
| `AllCards.json` | Card data from MTGJSON (v4, 2019) |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the data

```bash
pip install -r requirements.txt sentence-transformers gradio
python generate_model.py
```

This rewrites `data/`, then starts the Gradio demo; stop it with Ctrl+C.

## Deploy to Streamlit Community Cloud

1. Push the repository to GitHub, including the `data/` folder.
2. On [share.streamlit.io](https://share.streamlit.io), create an app from this repository:
   branch `main`, main file path `app.py`.
3. The first load takes a minute or two: it installs packages, downloads the ONNX model (~80 MB),
   and loads ChromaDB.

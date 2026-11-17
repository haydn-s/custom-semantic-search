"""MTG Card Search: a Streamlit app for semantic search over Magic: The Gathering cards.

At startup it loads the chunks and embeddings saved by generate_model.py (in data/)
into an in-memory ChromaDB collection. Queries are embedded with the same model,
all-MiniLM-L6-v2, using ChromaDB's built-in ONNX copy of it, so PyTorch isn't needed.

Run locally:  streamlit run app.py
"""

import gzip
import json
import re
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np
import streamlit as st
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DATA_DIR = Path(__file__).parent / "data"
NUM_CARDS = 5
NUM_RULINGS = 3
COLORS = {"White": "W", "Blue": "U", "Black": "B", "Red": "R", "Green": "G"}
FORMATS = ["Standard", "Pioneer", "Modern", "Pauper", "Legacy", "Vintage", "Commander"]
MAX_COSTS = [str(n) for n in range(11)] + ["Any"]
EXAMPLES = [
    "tap to add green mana elf",
    "counter target spell",
    "flying angel with lifelink",
    "draw a card when a creature enters the battlefield",
    "can I respond to Grim Lavamancer's exile cost?",
]

st.set_page_config(page_title="MTG Card Search", page_icon="🃏", layout="wide")


@st.cache_resource(show_spinner="Loading cards and rulings into ChromaDB...")
def load_collection():
    """Build the ChromaDB collection once per server process, from the files in data/.

    Returns the collection and how many chunks of each kind ("card", "ruling") it holds.
    """
    embeddings = np.load(DATA_DIR / "embeddings.npy")
    client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection("mtg_cards", embedding_function=DefaultEmbeddingFunction())

    # Read the chunks a batch at a time so they're never all in memory at once.
    batch_size = client.get_max_batch_size()
    batch, start, counts = [], 0, Counter()
    with gzip.open(DATA_DIR / "chunks.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            counts[chunk["metadata"]["kind"]] += 1
            batch.append(chunk)
            if len(batch) == batch_size:
                _upsert(collection, batch, embeddings[start:start + len(batch)])
                start += len(batch)
                batch = []
    if batch:
        _upsert(collection, batch, embeddings[start:])
    return collection, counts


def _upsert(collection, chunks, embeddings):
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["document"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings.astype(np.float32),
    )


def build_where(kind, colors, max_cost, fmt):
    """Turn the sidebar filters into a ChromaDB `where` clause."""
    conditions = [{"kind": kind}]
    conditions += [{f"color_{COLORS[color]}": True} for color in colors]
    if max_cost != "Any":
        conditions.append({"converted_mana_cost": {"$lte": float(max_cost)}})
    if fmt != "Any format":
        conditions.append({f"legality_{fmt.lower()}": {"$in": ["Legal", "Restricted"]}})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def search(collection, query, kind, n_results, where):
    results = collection.query(query_texts=[query], n_results=n_results, where=where)
    return list(zip(results["documents"][0], results["metadatas"][0]))


def md(text):
    """Escape Markdown so card text such as '*/*' or '+1:' shows literally, and keep line breaks."""
    return re.sub(r"([\\`*_\[\]#+\-!|<>$~])", r"\\\1", text).replace("\n", "  \n")


def show_card(meta):
    with st.container(border=True):
        cost = f" `{meta['mana_cost']}`" if meta["mana_cost"] else ""
        st.markdown(f"**{md(meta['name'])}**{cost}")
        details = [meta["type_line"]]
        if "power" in meta:
            details.append(f"{meta['power']}/{meta['toughness']}")
        if "loyalty" in meta:
            details.append(f"Loyalty {meta['loyalty']}")
        st.caption(md(" · ".join(details)))
        if meta["text"]:
            st.markdown(md(meta["text"]))


def show_ruling(document, meta):
    # Ruling documents look like "Ruling for <name> (<date>): <text>"
    prefix = f"Ruling for {meta['name']} ({meta['ruling_date']}): "
    text = document[len(prefix):] if document.startswith(prefix) else document
    with st.container(border=True):
        st.markdown(f"**{md(meta['name'])}**")
        st.caption(meta["ruling_date"])
        st.markdown(md(text))


def use_example():
    if st.session_state.example:
        st.session_state.query = st.session_state.example
    st.session_state.example = None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("🃏 MTG Card Search")
collection, counts = load_collection()
st.write(f"Describe a card in plain English to find the closest matches among {counts['card']:,} "
         f"Magic: The Gathering cards and {counts['ruling']:,} official rulings.")

with st.sidebar:
    st.header("Filters")
    colors = st.multiselect("Card includes colors", list(COLORS))
    max_cost = st.select_slider("Max converted mana cost", options=MAX_COSTS, value="Any")
    fmt = st.selectbox("Legal in", ["Any format", *FORMATS])
    st.caption("Filters apply to cards and rulings.")

query = st.text_input("Search", key="query", placeholder="e.g. destroy target creature",
                      label_visibility="collapsed")
st.pills("Try an example", EXAMPLES, key="example", on_change=use_example)

if query:
    cards_column, rulings_column = st.columns([3, 2], gap="large")

    with cards_column:
        st.subheader("Cards")
        cards = search(collection, query, "card", NUM_CARDS, build_where("card", colors, max_cost, fmt))
        if not cards:
            st.info("No cards match these filters.")
        for _, meta in cards:
            show_card(meta)

    with rulings_column:
        st.subheader("Rulings")
        rulings = search(collection, query, "ruling", NUM_RULINGS, build_where("ruling", colors, max_cost, fmt))
        if not rulings:
            st.info("No rulings match these filters.")
        for document, meta in rulings:
            show_ruling(document, meta)

st.divider()
st.caption("Card data: MTGJSON (2019) · Embeddings: all-MiniLM-L6-v2 · Vector database: ChromaDB")

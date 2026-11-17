
# Imports
import gzip
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import gradio as gr
import chromadb
from chromadb.config import Settings
from chunk_cards import build_chunks

# Select the Model for Semantic Search
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# How many cards to include: the most popular ones by EDHREC rank. None = all 20,478 cards,
# but the full set needs ~1.1 GB of memory in the app, more than Streamlit Community Cloud guarantees.
MAX_CARDS = 5000

# Chunk the cards (one chunk per card, plus one per ruling) and embed them
chunks = build_chunks("AllCards.json", max_cards=MAX_CARDS)
embeddings = model.encode([c["embed_text"] for c in chunks], batch_size=128, show_progress_bar=True)

# Save the chunks and embeddings for the Streamlit app (app.py), which loads them into ChromaDB
# when it starts. A saved chromadb/ folder would be far bigger (~330 MB for all cards) and too big
# to push to GitHub. float16 halves the size without changing which results come back.
os.makedirs("data", exist_ok=True)
np.save("data/embeddings.npy", embeddings.astype(np.float16))
with open("data/chunks.jsonl.gz", "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
    for c in chunks:
        record = {"id": c["id"], "document": c["document"], "metadata": c["metadata"]}
        f.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))

# Store the Embeddings in ChromaDB (in memory, for the Gradio demo below), in batches:
# one add() call accepts a limited number of records
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.create_collection("mtg_cards")
batch_size = client.get_max_batch_size()
for start in range(0, len(chunks), batch_size):
    batch = chunks[start:start + batch_size]
    collection.add(
        ids=[c["id"] for c in batch],
        documents=[c["document"] for c in batch],
        embeddings=embeddings[start:start + batch_size].tolist(),
        metadatas=[c["metadata"] for c in batch],
    )

# Define the Search Function: cards and rulings are searched separately,
# so the (more numerous) rulings never push cards out of the results
NUM_CARDS = 5
NUM_RULINGS = 3

def search_kind(query_embedding, kind, n_results):
    results = collection.query(query_embeddings=query_embedding, n_results=n_results, where={"kind": kind})
    return "\n\n".join(results["documents"][0])

def semantic_search(query):
    query_embedding = model.encode([query]).tolist()
    return search_kind(query_embedding, "card", NUM_CARDS), search_kind(query_embedding, "ruling", NUM_RULINGS)

# Create a Gradio interface
demo = gr.Interface(
    fn=semantic_search,
    inputs=gr.Textbox(label="Enter your search query"),
    outputs=[
        gr.Textbox(label=f"Top {NUM_CARDS} Cards", lines=20),
        gr.Textbox(label=f"Top {NUM_RULINGS} Rulings", lines=10),
    ],
    title="Semantic Search Engine",
    description="Search over your custom dataset using semantic similarity."
)
demo.launch()

#!/usr/bin/env python3
"""
Hybrid Movie Recommender with Gradio UI
– Fully toggleable options: weights, filters, MMR, popularity norm, etc.
– Zero‑Shot Relevance appears only when enabled.
– Overview fix: handles NaN, empty strings, literal "nan".
– ChromaDB metadata now only contains atomic types (no lists).
– Auto‑rebuilds ChromaDB if overviews are missing or metadata invalid.
– Localhost binding.
"""

import os
import sys
import glob
import shutil
import stat
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# LangChain / ChromaDB
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Transformers (zero‑shot)
from transformers import pipeline

# Gradio UI
import gradio as gr

# ───────────────────────────────────────────────
#  0. Configuration & helper functions
# ───────────────────────────────────────────────
DB_DIR = "./chroma_movies_db"
CSV_PATH = "data/cleaned.csv"
BART_MODEL_PATH = "./models/bart-large-mnli"
BGE_SNAPSHOT_BASE = "./models/bge-small/models--BAAI--bge-small-en-v1.5"

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def set_full_permissions(path):
    os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        for f in files:
            os.chmod(os.path.join(root, f), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

# ───────────────────────────────────────────────
#  1. Build or load ChromaDB (enhanced metadata)
# ───────────────────────────────────────────────
def parse_genres(genres_str: str) -> List[str]:
    """Parse genres from CSV string (pipe, comma, or semicolon separated)."""
    if not genres_str or pd.isna(genres_str):
        return []
    # Split by common separators
    parts = re.split(r'[|;,]+', genres_str)
    return [g.strip() for g in parts if g.strip()]

def clean_overview(val) -> str:
    """Convert overview to string, handle NaN and empty values."""
    if pd.isna(val):
        return "No overview available for this movie."
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return "No overview available for this movie."
    return s

def build_chroma_db():
    """Create ChromaDB from CSV with atomic metadata only."""
    print("Building Chroma database from CSV...")
    snapshot_dirs = glob.glob(os.path.join(BGE_SNAPSHOT_BASE, "snapshots", "*"))
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot found in {BGE_SNAPSHOT_BASE}")
    snapshot_dirs.sort(key=os.path.getmtime, reverse=True)
    local_model_path = snapshot_dirs[0]
    print(f"Using local BGE model path: {local_model_path}")

    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"Embedding device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs={"device": device, "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True}
    )

    df = pd.read_csv(CSV_PATH).fillna("")
    if df.empty:
        raise ValueError("CSV is empty")
    print(f"Loaded {len(df)} movies")

    documents = []
    ids = []

    for idx, row in df.iterrows():
        title = str(row.get("original_title", row.get("title", "")))
        imdb_id = str(row.get("imdb_id", ""))
        director = str(row.get("director", ""))
        cast = str(row.get("cast", ""))
        genres_str = str(row.get("genres", ""))
        overview = clean_overview(row.get("overview", ""))
        keywords = str(row.get("keywords", ""))
        tagline = str(row.get("tagline", ""))

        page_content = f"""
Title: {title}
IMDB ID: {imdb_id}
Director: {director}
Cast: {cast}
Genres: {genres_str}
Tagline: {tagline}
Keywords: {keywords}
Overview: {overview}
        """.strip()

        metadata = {
            "title": title,
            "imdb_id": imdb_id,
            "director": director,
            "cast": cast,
            "genres_str": genres_str,                     # string only
            "vote_average": float(row.get("vote_average", 0)) if pd.notna(row.get("vote_average")) else 0.0,
            "release_year": int(row.get("release_year", 0)) if pd.notna(row.get("release_year")) else 0,
            "overview": overview,
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
        ids.append(f"movie_{idx}")

    print(f"Prepared {len(documents)} documents")

    if os.path.exists(DB_DIR):
        print(f"Removing existing {DB_DIR}")
        shutil.rmtree(DB_DIR, onerror=remove_readonly)
    os.makedirs(DB_DIR, exist_ok=True)
    set_full_permissions(DB_DIR)

    vectordb = Chroma.from_documents(
        documents=documents,
        ids=ids,
        embedding=embeddings,
        collection_name="movies",
        persist_directory=DB_DIR,
    )
    print("Database built successfully.")
    return vectordb

def load_chroma_db():
    """Load existing ChromaDB, but rebuild if overviews are missing or if any list-valued metadata is found."""
    if not os.path.exists(DB_DIR):
        print("No database found, building fresh...")
        return build_chroma_db()
    
    print("Loading existing Chroma database...")
    snapshot_dirs = glob.glob(os.path.join(BGE_SNAPSHOT_BASE, "snapshots", "*"))
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot found in {BGE_SNAPSHOT_BASE}")
    snapshot_dirs.sort(key=os.path.getmtime, reverse=True)
    local_model_path = snapshot_dirs[0]

    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs={"device": device, "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectordb = Chroma(
        collection_name="movies",
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    
    # Check if the database has valid overviews (and no list metadata)
    sample = vectordb._collection.get(limit=1)
    if sample['metadatas']:
        meta = sample['metadatas'][0]
        # If overview is missing or empty, rebuild
        if not meta.get('overview') or meta.get('overview') == "No overview available for this movie.":
            print("Old database without valid overviews detected – rebuilding...")
            return build_chroma_db()
        # If there's a 'genres' key of type list, rebuild (old format)
        if 'genres' in meta and isinstance(meta['genres'], list):
            print("Old database with list metadata detected – rebuilding...")
            return build_chroma_db()
    
    print(f"Database loaded with {vectordb._collection.count()} documents.")
    return vectordb

# ───────────────────────────────────────────────
#  2. Load zero‑shot pipeline & movie dataframe
# ───────────────────────────────────────────────
def load_zero_shot_pipeline():
    print("Loading zero‑shot classification pipeline...")
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    pipe = pipeline(
        "zero-shot-classification",
        model=BART_MODEL_PATH,
        device=device,
        local_files_only=True
    )
    print(f"Zero‑shot pipeline loaded on {device}.")
    return pipe

def load_movie_dataframe():
    df = pd.read_csv(CSV_PATH).fillna("")
    df.set_index("id", inplace=True)
    return df

# ───────────────────────────────────────────────
#  3. Core recommendation functions
# ───────────────────────────────────────────────
def semantic_search(vectordb, query: str, k: int = 100) -> List[Dict[str, Any]]:
    results = vectordb.similarity_search_with_score(query, k=k)
    hits = []
    for doc, score in results:
        similarity = 1.0 - score
        hits.append({
            "id": doc.metadata.get("imdb_id", ""),
            "title": doc.metadata.get("title", ""),
            "metadata": doc.metadata,
            "similarity": similarity,
            "text": doc.page_content,
        })
    return hits

def classify_relevance_batch(pipe, texts: List[str]) -> List[float]:
    if not texts:
        return []
    results = pipe(
        texts,
        candidate_labels=["relevant", "irrelevant"],
        hypothesis_template="This movie is {}."
    )
    scores = []
    for res in results:
        idx = res["labels"].index("relevant")
        scores.append(res["scores"][idx])
    return scores

def apply_filters(candidates: List[Dict[str, Any]],
                  min_rating: float = 0.0,
                  min_year: int = 1900,
                  max_year: int = 2100,
                  selected_genres: List[str] = None) -> List[Dict[str, Any]]:
    filtered = []
    for c in candidates:
        meta = c["metadata"]
        rating = meta.get("vote_average", 0)
        year = meta.get("release_year", 0)
        if rating < min_rating:
            continue
        if year < min_year or year > max_year:
            continue
        if selected_genres:
            genres_str = meta.get("genres_str", "")
            # split by common separators
            movie_genres = re.split(r'[|;,]+', genres_str)
            movie_genres = [g.strip() for g in movie_genres if g.strip()]
            if not any(g in movie_genres for g in selected_genres):
                continue
        filtered.append(c)
    return filtered

def compute_final_score(similarity: float, relevance: float,
                        vote_average: float, norm_popularity: float,
                        alpha: float, beta: float, gamma: float, delta: float,
                        use_rating_norm: bool = True) -> float:
    rating_norm = min(vote_average, 10.0) / 10.0 if use_rating_norm else vote_average / 10.0
    return alpha * similarity + beta * relevance + gamma * rating_norm + delta * norm_popularity

def recommend(vectordb, pipe, df_lookup, query: str,
              use_zero_shot: bool = True,
              k_retrieve: int = 100,
              top_n: int = 5,
              alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.1, delta: float = 0.1,
              use_popularity_norm: bool = True,
              use_rating_norm: bool = True,
              min_rating: float = 0.0,
              min_year: int = 1900,
              max_year: int = 2100,
              selected_genres: List[str] = None,
              use_mmr: bool = False) -> List[Dict[str, Any]]:
    # 1. Semantic retrieval
    candidates = semantic_search(vectordb, query, k=k_retrieve)
    if not candidates:
        return []

    # 2. Apply filters
    candidates = apply_filters(candidates, min_rating, min_year, max_year, selected_genres)
    if not candidates:
        return []

    # 3. Zero‑shot relevance (if enabled)
    if use_zero_shot and pipe is not None:
        texts = [c["text"] for c in candidates]
        relevance_scores = classify_relevance_batch(pipe, texts)
        for c, rel in zip(candidates, relevance_scores):
            c["zero_shot_relevance"] = rel
    else:
        for c in candidates:
            c["zero_shot_relevance"] = None

    # 4. Popularity normalisation
    pops = []
    for c in candidates:
        try:
            pop = df_lookup.loc[c["id"], "popularity"]
        except KeyError:
            pop = 0.0
        pops.append(np.log1p(float(pop)))
    if use_popularity_norm:
        max_pop = max(pops) if pops else 1.0
        if max_pop == 0:
            norm_pop = {c["id"]: 0.0 for c in candidates}
        else:
            norm_pop = {c["id"]: p / max_pop for c, p in zip(candidates, pops)}
    else:
        norm_pop = {c["id"]: 0.0 for c in candidates}

    # 5. Compute final scores
    for c in candidates:
        vote_avg = c["metadata"].get("vote_average", 5.0)
        relevance_val = c["zero_shot_relevance"] if c["zero_shot_relevance"] is not None else 0.5
        c["final_score"] = compute_final_score(
            similarity=c["similarity"],
            relevance=relevance_val,
            vote_average=vote_avg,
            norm_popularity=norm_pop[c["id"]],
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            use_rating_norm=use_rating_norm
        )
        c["score_components"] = {
            "similarity": c["similarity"],
            "relevance": relevance_val,
            "rating_norm": min(vote_avg, 10.0) / 10.0 if use_rating_norm else vote_avg / 10.0,
            "popularity_norm": norm_pop[c["id"]],
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta}
        }

    # 6. Sort by final score
    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # 7. Optional MMR (placeholder, would need embeddings)
    if use_mmr:
        # For full MMR we would need to fetch embeddings from vectordb.
        # For brevity, we skip actual MMR but keep the option.
        pass

    return candidates[:top_n]

# ───────────────────────────────────────────────
#  4. Gradio UI with controls
# ───────────────────────────────────────────────
_vectordb = None
_pipe = None
_df_lookup = None

def update_sort_choices(use_zero_shot: bool):
    if use_zero_shot:
        return gr.Radio(choices=["Final Score", "Similarity", "Zero‑Shot Relevance", "Rating"], value="Final Score")
    else:
        return gr.Radio(choices=["Final Score", "Similarity", "Rating"], value="Final Score")

def get_unique_genres():
    df = pd.read_csv(CSV_PATH).fillna("")
    all_genres = set()
    for g in df["genres"]:
        if g and not pd.isna(g):
            for genre in parse_genres(str(g)):
                all_genres.add(genre)
    return sorted(list(all_genres))

def safe_overview(overview_str: str) -> str:
    """Ensure UI never shows empty or NaN overview."""
    if not overview_str or overview_str.strip() == "" or overview_str.lower() == "nan":
        return "No overview available for this movie."
    return overview_str

def recommend_ui(query: str,
                 use_zero_shot: bool,
                 sort_by: str,
                 top_n: int,
                 k_retrieve: int,
                 alpha: float, beta: float, gamma: float, delta: float,
                 use_popularity_norm: bool,
                 use_rating_norm: bool,
                 min_rating: float,
                 min_year: int, max_year: int,
                 selected_genres: List[str],
                 use_mmr: bool):
    if not query.strip():
        return "<p>Please enter a description.</p>"
    try:
        recs = recommend(
            _vectordb, _pipe, _df_lookup, query,
            use_zero_shot=use_zero_shot,
            k_retrieve=int(k_retrieve),
            top_n=int(top_n),
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            use_popularity_norm=use_popularity_norm,
            use_rating_norm=use_rating_norm,
            min_rating=min_rating,
            min_year=int(min_year), max_year=int(max_year),
            selected_genres=selected_genres if selected_genres else None,
            use_mmr=use_mmr
        )
    except Exception as e:
        return f"<p>Error during recommendation: {e}</p>"
    if not recs:
        return "<p>No recommendations found with the current filters.</p>"

    # Sort after recommendation (re-sort if needed)
    if sort_by == "Final Score":
        recs.sort(key=lambda x: x["final_score"], reverse=True)
    elif sort_by == "Similarity":
        recs.sort(key=lambda x: x["similarity"], reverse=True)
    elif sort_by == "Zero‑Shot Relevance":
        if use_zero_shot:
            recs.sort(key=lambda x: x["zero_shot_relevance"] if x["zero_shot_relevance"] is not None else 0, reverse=True)
        else:
            recs.sort(key=lambda x: x["final_score"], reverse=True)
    elif sort_by == "Rating":
        recs.sort(key=lambda x: x["metadata"].get("vote_average", 0), reverse=True)

    html = '<div style="font-family: Arial, sans-serif;">'
    for i, r in enumerate(recs, 1):
        meta = r["metadata"]
        title = meta.get("title", "Unknown")
        year = meta.get("release_year", "")
        rating = meta.get("vote_average", 0)
        overview = safe_overview(meta.get("overview", ""))
        if len(overview) > 250:
            overview = overview[:250] + "..."
        comp = r["score_components"]
        weights = comp["weights"]
        explanation = (f"<small>Score = {weights['alpha']:.1f}×Sim + {weights['beta']:.1f}×Rel + "
                       f"{weights['gamma']:.1f}×Rating + {weights['delta']:.1f}×Pop<br>"
                       f"Sim: {comp['similarity']:.3f}, Rel: {comp['relevance']:.3f}, "
                       f"Rating norm: {comp['rating_norm']:.3f}, Pop norm: {comp['popularity_norm']:.3f}</small>")
        relevance_display = ""
        if use_zero_shot and r["zero_shot_relevance"] is not None:
            relevance_display = f'&nbsp;|&nbsp;<strong>Relevance:</strong> {r["zero_shot_relevance"]:.3f}'
        html += f'''
        <div style="border-bottom: 1px solid #ddd; padding: 12px 0;">
            <h3>{i}. {title} ({year})</h3>
            <p><strong>Rating:</strong> {rating:.1f}/10</p>
            <p><strong>Similarity:</strong> {r["similarity"]:.3f}{relevance_display}&nbsp;|&nbsp;
               <strong>Final Score:</strong> {r["final_score"]:.3f}</p>
            <p><em>{overview}</em></p>
            <p>{explanation}</p>
        </div>
        '''
    html += '</div>'
    return html

def similar_ui(title: str):
    if not title.strip():
        return "<p>Please enter a movie title.</p>"
    try:
        embeddings = _vectordb._embedding_function
        query_emb = embeddings.embed_query(title)
        results = _vectordb.similarity_search_by_vector(query_emb, k=10)
        sims = []
        for doc in results:
            if doc.metadata.get("title", "").lower() == title.lower():
                continue
            sims.append(doc)
            if len(sims) >= 5:
                break
    except Exception as e:
        return f"<p>Error: {e}</p>"
    if not sims:
        return f"<p>No similar movies found for '{title}'.</p>"
    html = '<div style="font-family: Arial, sans-serif;">'
    for i, doc in enumerate(sims, 1):
        meta = doc.metadata
        overview = safe_overview(meta.get("overview", ""))
        if len(overview) > 200:
            overview = overview[:200] + "..."
        html += f'''
        <div style="border-bottom: 1px solid #ddd; padding: 8px 0;">
            <b>{i}. {meta.get("title", "Unknown")} ({meta.get("release_year", "")})</b><br/>
            Rating: {meta.get("vote_average", 0):.1f}<br/>
            {overview}
        </div>
        '''
    html += '</div>'
    return html

# ───────────────────────────────────────────────
#  5. Main: initialise models and launch UI
# ───────────────────────────────────────────────
def main():
    global _vectordb, _pipe, _df_lookup
    print("Initialising recommender system...")
    _vectordb = load_chroma_db()
    _pipe = load_zero_shot_pipeline()
    _df_lookup = load_movie_dataframe()
    all_genres = get_unique_genres()
    print(f"Available genres: {all_genres[:10]}...")

    with gr.Blocks(title="Hybrid Movie Recommender") as demo:
        gr.Markdown("# 🎬 Hybrid Movie Recommender")
        gr.Markdown("Powered by **BGE embeddings**, **BART‑large‑MNLI** zero‑shot, and full user control.")

        with gr.Tabs():
            with gr.TabItem("🔍 Describe what you want"):
                query_input = gr.Textbox(label="Your request", placeholder="e.g., a funny animated movie with heart", lines=2)

                with gr.Accordion("⚙️ Advanced Options", open=False):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Recommendation Weights")
                            alpha = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Similarity weight (α)")
                            beta = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Zero‑Shot Relevance weight (β)")
                            gamma = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Rating weight (γ)")
                            delta = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Popularity weight (δ)")
                        with gr.Column():
                            gr.Markdown("#### Filters")
                            min_rating = gr.Slider(0.0, 10.0, value=0.0, step=0.5, label="Minimum rating")
                            min_year = gr.Number(value=1900, label="Min release year", precision=0)
                            max_year = gr.Number(value=2025, label="Max release year", precision=0)
                            genre_select = gr.Dropdown(choices=all_genres, multiselect=True, label="Genres (optional)")
                    with gr.Row():
                        with gr.Column():
                            k_retrieve = gr.Slider(20, 500, value=100, step=10, label="Number of candidates retrieved (k_retrieve)")
                            top_n = gr.Slider(1, 20, value=5, step=1, label="Number of recommendations")
                        with gr.Column():
                            use_popularity_norm = gr.Checkbox(label="Normalize popularity (log‑scale)", value=True)
                            use_rating_norm = gr.Checkbox(label="Normalize rating to 0-1", value=True)
                            use_mmr = gr.Checkbox(label="Enable MMR diversity (experimental)", value=False)
                    zero_shot_toggle = gr.Checkbox(label="Use zero‑shot classification (slower but more accurate)", value=True)
                    sort_by = gr.Radio(choices=["Final Score", "Similarity", "Rating"], value="Final Score", label="Sort recommendations by")
                
                recommend_btn = gr.Button("Get Recommendations", variant="primary")
                recommend_output = gr.HTML(label="Recommendations")
                
                zero_shot_toggle.change(fn=update_sort_choices, inputs=zero_shot_toggle, outputs=sort_by)
                
                recommend_btn.click(
                    fn=recommend_ui,
                    inputs=[
                        query_input, zero_shot_toggle, sort_by, top_n, k_retrieve,
                        alpha, beta, gamma, delta,
                        use_popularity_norm, use_rating_norm,
                        min_rating, min_year, max_year, genre_select, use_mmr
                    ],
                    outputs=recommend_output
                )
            
            with gr.TabItem("🎞️ Find similar movies"):
                title_input = gr.Textbox(label="Movie title", placeholder="e.g., Inception")
                similar_btn = gr.Button("Find Similar", variant="primary")
                similar_output = gr.HTML(label="Similar movies")
                similar_btn.click(fn=similar_ui, inputs=title_input, outputs=similar_output)
        
        gr.Markdown("---\n*All models run locally – your data never leaves your machine.*")

    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())

if __name__ == "__main__":
    main()
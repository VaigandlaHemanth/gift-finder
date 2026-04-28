"""
Catalog loader and vector search for the Mumzworld gift finder.

The catalog is intentionally local: reviewers only need a Groq key for the
LLM, while retrieval runs with sentence-transformers + ChromaDB on their
machine.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mumzworld_products"
PERSIST_DIR = "chroma_db"

_embedding_model: Any | None = None
_chroma_client: Any | None = None
_loaded_catalog_path: str | None = None


def _get_embedding_model() -> Any:
    """Load the embedding model only when retrieval is first used."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        print("Loading embedding model...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_chroma_client():
    """Create a Chroma client compatible with current and older versions."""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    import chromadb

    if hasattr(chromadb, "PersistentClient"):
        _chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    else:
        from chromadb.config import Settings

        _chroma_client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR)
        )
    return _chroma_client


def get_or_create_collection():
    """Get or create the products collection using cosine distance."""
    client = _get_chroma_client()
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def _product_document(product: dict[str, Any]) -> str:
    """Build one bilingual retrieval document for a product."""
    tags = ", ".join(product.get("tags", []))
    return (
        f"{product['name_en']}. {product['category_en']}. {product['description_en']} "
        f"Price: {product['price_aed']} AED. "
        f"Age: {product['age_months_min']}-{product['age_months_max']} months. "
        f"Tags: {tags}. "
        f"{product['name_ar']}. {product['category_ar']}. {product['description_ar']}"
    )


def load_products_to_chromadb(products_path: str = "data/products.json", force: bool = False):
    """
    Load products into ChromaDB.

    Re-loading is skipped within the same process unless `force=True`, which
    keeps the UI responsive after the first request.
    """
    global _loaded_catalog_path

    resolved_path = str(Path(products_path).resolve())
    collection = get_or_create_collection()

    if _loaded_catalog_path == resolved_path and not force:
        return collection

    with open(products_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    try:
        existing = collection.get()
        if existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    documents = []
    metadatas = []
    ids = []

    for product in products:
        documents.append(_product_document(product))
        metadatas.append(
            {
                "id": product["id"],
                "name_en": product["name_en"],
                "name_ar": product["name_ar"],
                "category_en": product["category_en"],
                "category_ar": product["category_ar"],
                "price_aed": product["price_aed"],
                "age_min": product["age_months_min"],
                "age_max": product["age_months_max"],
                "tags": json.dumps(product["tags"], ensure_ascii=False),
                "description_en": product["description_en"],
                "description_ar": product["description_ar"],
                "avg_rating": product["avg_rating"],
                "num_reviews": product["num_reviews"],
                "in_stock": product["in_stock"],
                "brand": product["brand"],
            }
        )
        ids.append(product["id"])

    print(f"Generating embeddings for {len(documents)} products...")
    embeddings = _get_embedding_model().encode(documents, show_progress_bar=True)

    collection.add(
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    _loaded_catalog_path = resolved_path
    print(f"Loaded {len(products)} products into ChromaDB")
    return collection


def search_products(query: str, n_results: int = 10, products_path: str = "data/products.json"):
    """
    Search products by semantic similarity.

    Returns product metadata with a `similarity` value where higher is better.
    """
    if not query or not query.strip():
        return []

    load_products_to_chromadb(products_path)
    collection = get_or_create_collection()
    query_embedding = _get_embedding_model().encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    products_with_scores = []
    for index, metadata in enumerate(results["metadatas"][0]):
        distance = results["distances"][0][index]
        similarity = max(0.0, min(1.0, 1 - float(distance)))
        products_with_scores.append(
            {
                **metadata,
                "similarity": similarity,
                "tags": json.loads(metadata["tags"]),
            }
        )

    return products_with_scores


if __name__ == "__main__":
    load_products_to_chromadb(force=True)

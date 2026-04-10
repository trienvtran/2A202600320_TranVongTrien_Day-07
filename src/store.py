from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # Initialize chromadb client + collection
            self._chroma_client = chromadb.Client()
            self._collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        return {
            'doc_id': doc.id,
            'content': doc.content,
            'metadata': doc.metadata,
            'embedding': self._embedding_fn(doc.content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run in-memory similarity search over provided records."""
        query_embedding = self._embedding_fn(query)
        
        # Compute similarity scores
        scored_records = []
        for record in records:
            doc_embedding = record['embedding']
            score = _dot(query_embedding, doc_embedding)
            scored_records.append({
                **record,
                'score': score
            })
        
        # Sort by score descending and return top_k
        scored_records.sort(key=lambda x: x['score'], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma:
            # Add to ChromaDB
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for doc in docs:
                record_id = f"{doc.id}_{self._next_index}"
                self._next_index += 1
                ids.append(record_id)
                documents.append(doc.content)
                embeddings.append(self._embedding_fn(doc.content))
                metadatas.append({**doc.metadata, 'doc_id': doc.id})
            
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            # Add to in-memory store
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)
                self._next_index += 1

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format ChromaDB results
            output = []
            if results['documents'] and len(results['documents']) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    output.append({
                        'doc_id': results['metadatas'][0][i].get('doc_id', ''),
                        'content': doc,
                        'score': results['distances'][0][i] if 'distances' in results else 0.0,
                        'metadata': results['metadatas'][0][i]
                    })
            return output
        else:
            # In-memory search
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k)
        
        if self._use_chroma:
            query_embedding = self._embedding_fn(query)
            # Build where filter for ChromaDB
            where_filter = metadata_filter
            results = self._collection.query(
                query_embeddings=[query_embedding],
                where=where_filter,
                n_results=top_k
            )
            
            # Format results
            output = []
            if results['documents'] and len(results['documents']) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    output.append({
                        'doc_id': results['metadatas'][0][i].get('doc_id', ''),
                        'content': doc,
                        'score': results['distances'][0][i] if 'distances' in results else 0.0,
                        'metadata': results['metadatas'][0][i]
                    })
            return output
        else:
            # In-memory: filter first, then search
            filtered_records = []
            for record in self._store:
                # Check if all filter conditions match
                match = True
                for key, value in metadata_filter.items():
                    if record['metadata'].get(key) != value:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
            
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            # Delete from ChromaDB
            try:
                # Get all records with this doc_id
                results = self._collection.get(
                    where={'doc_id': doc_id}
                )
                if results['ids']:
                    self._collection.delete(ids=results['ids'])
                    return True
                return False
            except Exception:
                return False
        else:
            # Delete from in-memory store
            initial_len = len(self._store)
            self._store = [r for r in self._store if r['doc_id'] != doc_id]
            return len(self._store) < initial_len

import os
import faiss
import pickle
import numpy as np
import hashlib
from pathlib import Path
from sentence_transformers import SentenceTransformer


class KnowledgeIndex:
    def __init__(self, vault_dir, embed_model="all-MiniLM-L6-v2", chunk_size=500, verbose=False):
        self.vault_dir = Path(vault_dir).expanduser()
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.embedder = SentenceTransformer(embed_model)
        self.index = None
        self.documents = []
        self.index_file = self.vault_dir / "knowledge.index.faiss"
        self.meta_file = self.vault_dir / "knowledge.index.meta"
        self.files_hash_file = self.vault_dir / "knowledge.index.hash"

    def _log(self, msg):
        if self.verbose:
            print(f"[KnowledgeIndex] {msg}")

    def _hash_vault_files(self):
        """Compute a hash of all markdown files in the vault to detect changes."""
        m = hashlib.md5()
        for root, _, files in os.walk(self.vault_dir):
            for f in sorted(files):
                if f.endswith(".md"):
                    path = Path(root) / f
                    m.update(str(path).encode())
                    m.update(str(os.path.getmtime(path)).encode())
        return m.hexdigest()

    def _save_hash(self, hash_val):
        with open(self.files_hash_file, "w") as f:
            f.write(hash_val)

    def _load_hash(self):
        if self.files_hash_file.exists():
            with open(self.files_hash_file, "r") as f:
                return f.read().strip()
        return None

    def _chunk_text(self, text):
        words = text.split()
        return [" ".join(words[i:i + self.chunk_size])
                for i in range(0, len(words), self.chunk_size)]

    def _load_notes(self):
        notes = []
        for root, _, files in os.walk(self.vault_dir):
            for file in files:
                if file.endswith(".md"):
                    path = Path(root) / file
                    with open(path, "r", encoding="utf-8") as f:
                        notes.append(f.read())
        return notes

    def build_index(self):
        """Build or rebuild the FAISS index if vault changed."""
        current_hash = self._hash_vault_files()
        saved_hash = self._load_hash()

        if self.index_file.exists() and saved_hash == current_hash:
            self._log("No changes detected — loading existing index.")
            self.load_index()
            return

        self._log("Changes detected or index missing — rebuilding index.")
        notes = self._load_notes()
        chunks = []
        for note in notes:
            chunks.extend(self._chunk_text(note))

        if not chunks:
            self._log("No notes found to index.")
            return

        embeddings = self.embedder.encode(chunks, show_progress_bar=self.verbose)
        embeddings = np.array(embeddings).astype("float32")
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.documents = chunks

        # Save index and metadata
        faiss.write_index(self.index, str(self.index_file))
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.documents, f)
        self._save_hash(current_hash)

        self._log(f"Built index with {len(chunks)} chunks.")

    def load_index(self):
        if not self.index_file.exists() or not self.meta_file.exists():
            self._log("No existing index found.")
            return False

        self.index = faiss.read_index(str(self.index_file))
        with open(self.meta_file, "rb") as f:
            self.documents = pickle.load(f)
        self._log(f"Loaded index with {len(self.documents)} chunks.")
        return True

    def query(self, text, top_k=5):
        if self.index is None or self.index.ntotal == 0:
            self._log("Index not loaded or empty.")
            return []

        vec = self.embedder.encode([text], convert_to_numpy=True).astype("float32")
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        distances, indices = self.index.search(vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({"text": self.documents[idx], "score": float(dist)})
        return results


def load_or_build_index(config):
    """Load the index from disk or build a new one from the vault."""
    vault_dir = config.get("vault_dir", "./vault")
    ki = KnowledgeIndex(vault_dir=vault_dir,
                        embed_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
                        chunk_size=config.get("chunk_size", 500),
                        verbose=True)
    ki.build_index()
    return ki

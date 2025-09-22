"""
Cache and Storage Management for the ORION Medical AI System
"""

import os
import json
import logging
import hashlib
import pickle
import gc
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from collections import defaultdict
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class DiskBasedGlobalContextCache:
    def __init__(self, cache_dir="cache/global_context", max_cache_size_mb=500, cache_ttl_hours=168):  # 7 days default
        self.cache_dir = cache_dir
        self.max_cache_size_mb = max_cache_size_mb
        self.cache_ttl_hours = cache_ttl_hours
        self.memory_cache = {}  # Fast lookup cache

        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_existing_cache()
        logger.info(f"Disk-based global context cache initialized at {self.cache_dir}")

    def _generate_cache_key(self, series_uid: str, slice_index: int, image_hash: str) -> str:
        """Generate unique cache key."""
        key_data = f"{series_uid}_{slice_index}_{image_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_image_hash(self, slice_array: np.ndarray) -> str:
        """Generate content hash of the image."""
        # Downsample for hash calculation to improve performance
        small_img = slice_array[::8, ::8]  # Every 8th pixel
        normalized = ((small_img - small_img.min()) / (small_img.max() - small_img.min() + 1e-8) * 255).astype(np.uint8)
        return hashlib.md5(normalized.tobytes()).hexdigest()[:12]

    def _get_cache_file_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _load_existing_cache(self):
        """Load valid cache entries from disk."""
        loaded_count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                try:
                    cache_key = filename[:-5]
                    file_path = self._get_cache_file_path(cache_key)

                    with open(file_path, 'r') as f:
                        cache_entry = json.load(f)

                    if self._is_cache_valid(cache_entry):
                        self.memory_cache[cache_key] = {
                            'timestamp': cache_entry['timestamp'],
                            'series_uid': cache_entry['series_uid'],
                            'slice_index': cache_entry['slice_index']
                        }
                        loaded_count += 1
                    else:
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to load cache file {filename}: {e}")

        logger.info(f"Loaded {loaded_count} valid cache entries from disk")

    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        if 'timestamp' not in cache_entry:
            return False
        entry_time = datetime.fromisoformat(cache_entry['timestamp'])
        return datetime.now() < entry_time + timedelta(hours=self.cache_ttl_hours)

    def get_cached_context(self, series_uid: str, slice_index: int, slice_array: np.ndarray) -> Optional[Dict]:
        """Retrieve cached global context."""
        image_hash = self._get_image_hash(slice_array)
        cache_key = self._generate_cache_key(series_uid, slice_index, image_hash)

        # Check memory cache first
        if cache_key in self.memory_cache:
            try:
                file_path = self._get_cache_file_path(cache_key)
                with open(file_path, 'r') as f:
                    cache_entry = json.load(f)

                if self._is_cache_valid(cache_entry):
                    logger.info(f"Cache HIT for slice {slice_index} (series: {series_uid[:8]}...)")
                    return cache_entry['data']
                else:
                    # Clean up expired cache
                    del self.memory_cache[cache_key]
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}")
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]

        logger.info(f"Cache MISS for slice {slice_index}")
        return None

    def store_context(self, series_uid: str, slice_index: int, slice_array: np.ndarray, context_data: Dict):
        """Store global context to disk cache."""
        image_hash = self._get_image_hash(slice_array)
        cache_key = self._generate_cache_key(series_uid, slice_index, image_hash)

        cache_entry = {
            'data': context_data,
            'timestamp': datetime.now().isoformat(),
            'series_uid': series_uid,
            'slice_index': slice_index,
            'image_hash': image_hash
        }

        try:
            # Store to disk
            file_path = self._get_cache_file_path(cache_key)
            with open(file_path, 'w') as f:
                json.dump(cache_entry, f, indent=2)

            # Update memory cache
            self.memory_cache[cache_key] = {
                'timestamp': cache_entry['timestamp'],
                'series_uid': series_uid,
                'slice_index': slice_index
            }

            self._cleanup_old_cache()
            logger.info(f"Stored global context for slice {slice_index} to disk cache")

        except Exception as e:
            logger.error(f"Failed to store cache: {e}")

    def _cleanup_old_cache(self):
        """Remove old cache files to stay within size limits."""
        try:
            cache_files = [(f, os.path.getmtime(os.path.join(self.cache_dir, f)))
                          for f in os.listdir(self.cache_dir) if f.endswith('.json')]

            total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f, _ in cache_files) / (1024 * 1024)

            if total_size > self.max_cache_size_mb:
                cache_files.sort(key=lambda x: x[1])  # Sort by modification time

                for filename, _ in cache_files:
                    file_path = os.path.join(self.cache_dir, filename)
                    os.remove(file_path)

                    cache_key = filename[:-5]
                    if cache_key in self.memory_cache:
                        del self.memory_cache[cache_key]

                    # Recalculate size
                    total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f))
                                   for f in os.listdir(self.cache_dir) if f.endswith('.json')) / (1024 * 1024)

                    if total_size <= self.max_cache_size_mb:
                        break

                logger.info(f"Cache cleanup completed. New size: {total_size:.2f} MB")
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

    def clear_cache(self, series_uid: str = None):
        """Clear cache entries. If series_uid provided, only clear for that series."""
        if series_uid:
            # Clear specific series
            keys_to_remove = []
            for cache_key, cache_entry in self.memory_cache.items():
                try:
                    file_path = self._get_cache_file_path(cache_key)
                    with open(file_path, 'r') as f:
                        cache_data = json.load(f)
                    if cache_data.get('series_uid') == series_uid:
                        keys_to_remove.append(cache_key)
                except Exception:
                    continue

            for cache_key in keys_to_remove:
                del self.memory_cache[cache_key]
                file_path = self._get_cache_file_path(cache_key)
                if os.path.exists(file_path):
                    os.remove(file_path)

            logger.info(f"Cleared cache for series {series_uid[:8]}...")
        else:
            # Clear all cache
            self.memory_cache.clear()
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("Cleared all global context cache")


class VectorDBManager:
    def __init__(self, db_path="vector_db"):
        self.db_path, self.index_file, self.meta_file = db_path, os.path.join(db_path, "rag.index"), os.path.join(db_path, "rag.meta")
        self.dimension, self.index, self.metadata = 768, None, []
        os.makedirs(self.db_path, exist_ok=True)

    def load_database(self):
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
                logger.info(f"Loading vector database from {self.db_path}...")
                self.index = faiss.read_index(self.index_file)
                with open(self.meta_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Successfully loaded {self.index.ntotal} vectors.")
            else:
                logger.warning("No existing vector database found. Initializing a new one.")
                self.initialize_new_database()
        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
            self.initialize_new_database()

    def initialize_new_database(self):
        """Initialize a new FAISS index"""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.metadata = []
        logger.info("Initialized new vector database.")

    def add_context(self, context_data: Dict, embedding: np.ndarray):
        """Add a context and its embedding to the database"""
        if self.index is None:
            self.initialize_new_database()

        # Normalize embedding for cosine similarity
        normalized_embedding = embedding / np.linalg.norm(embedding)
        self.index.add(normalized_embedding.reshape(1, -1).astype(np.float32))
        self.metadata.append(context_data)

        # Save to disk
        self.save_database()

    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar contexts"""
        if self.index is None or self.index.ntotal == 0:
            return []

        # Normalize query embedding
        normalized_query = query_embedding / np.linalg.norm(query_embedding)

        # Search
        scores, indices = self.index.search(normalized_query.reshape(1, -1).astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)

        return results

    def save_database(self):
        """Save the vector database to disk"""
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.meta_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info("Vector database saved to disk.")
        except Exception as e:
            logger.error(f"Failed to save vector database: {e}")


class BackendCache:
    def __init__(self):
        self.global_context_cache = {}
        self.roi_analysis_cache = {}
        self.cache_lock = threading.Lock()

    def get_global_context(self, series_uid: str, slice_index: int):
        with self.cache_lock:
            cache_key = f"{series_uid}_{slice_index}"
            return self.global_context_cache.get(cache_key)

    def set_global_context(self, series_uid: str, slice_index: int, context):
        with self.cache_lock:
            cache_key = f"{series_uid}_{slice_index}"
            self.global_context_cache[cache_key] = context

    def clear_series_cache(self, series_uid: str):
        with self.cache_lock:
            keys_to_remove = [k for k in self.global_context_cache.keys() if k.startswith(f"{series_uid}_")]
            for key in keys_to_remove:
                del self.global_context_cache[key]

    def clear_all_cache(self):
        with self.cache_lock:
            self.global_context_cache.clear()
            self.roi_analysis_cache.clear()
            gc.collect()
            logger.info("All backend caches cleared")
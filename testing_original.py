import os
import json
import logging
import hashlib
import tempfile
import asyncio
import gc
import pickle
import math
import time
import threading
import uuid
from collections import defaultdict
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union, Literal, Optional, Tuple, NamedTuple
import psutil
import uvicorn
import pydicom
import numpy as np
import cv2
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Path, Body, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, validator, Field
from PIL import Image
from google.cloud import storage
import google.auth
import google.auth.transport.requests
from transformers import AutoProcessor, AutoModel
from pydantic import BaseModel, Field
from typing import Union, Literal
from Swin_LiteMedSAM.models.mask_decoder import MaskDecoder_Prompt
from Swin_LiteMedSAM.models.prompt_encoder import PromptEncoder
from Swin_LiteMedSAM.models.swin import SwinTransformer
from Swin_LiteMedSAM.models.transformer import TwoWayTransformer
from skimage.measure import regionprops
from pydantic import BaseModel, Field
from typing import Union, Literal
import faiss
from skimage import measure
from threading import Lock


try:
    import trimesh
except ImportError as e:
    print(f"Failed to import trimesh: {e}")
    trimesh = None

from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Ensures output to stdout/stderr
    ]
)

uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.INFO)


logger = logging.getLogger(__name__)

# Force logging to be unbuffered for Docker
import sys
sys.stdout.flush()
sys.stderr.flush()

CANVAS_WIDTH = 512
CANVAS_HEIGHT = 512
CANVAS_SIZE = (CANVAS_WIDTH, CANVAS_HEIGHT)

# Image processing configuration
IMAGE_TARGET_SIZE = (512, 512)  # Standard size for all image operations
COORDINATE_SCALE = 512  # Scale for converting between canvas and volume coordinates

# DICOM data directory configuration - portable across environments

#DICOM_DATA_ROOT = os.getenv("DICOM_DATA_ROOT", "/data") # production time!!

DICOM_DATA_ROOT = "/"  #for testing in the ORION directory
logger.info(f"DICOM data root configured to: {DICOM_DATA_ROOT}")

class CustomJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        def custom_serializer(obj):
            if isinstance(obj, (pydicom.multival.MultiValue, np.ndarray)): return list(obj)
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            try: return json.JSONEncoder().default(obj)
            except TypeError: return str(obj)
        return json.dumps(content, default=custom_serializer, indent=2).encode("utf-8")

app = FastAPI(
    title="Spatial-Priority Medical RAG System",
    version="14.0.0-enhanced",
    description="RAG system prioritizing spatial context for medical diagnosis, with disk caching and region-based ROI analysis.",
    default_response_class=CustomJSONResponse
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# METRICS TRACKING

metrics_lock = Lock()
disease_metrics = {
    "service_start_time": datetime.now(),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "model_predictions": 0,
    "dicom_files_processed": 0,
    "cache_operations": 0,
    "roi_analyses": 0,
    "response_times": [],
    "last_request_time": None,
    "active_sessions": 0,
    "memory_peak_usage": 0
}

# ROI REGION HANDLING CLASSES
class Point(NamedTuple):
    x: float
    y: float

class RectangularROI(NamedTuple):
    top_left: Point
    bottom_right: Point
    
    def contains_point(self, point: Point) -> bool:
        return (self.top_left.x <= point.x <= self.bottom_right.x and 
                self.top_left.y <= point.y <= self.bottom_right.y)
    
    def get_center(self) -> Point:
        return Point(
            (self.top_left.x + self.bottom_right.x) / 2,
            (self.top_left.y + self.bottom_right.y) / 2
        )
    
    def get_area(self) -> float:
        return abs(self.bottom_right.x - self.top_left.x) * abs(self.bottom_right.y - self.top_left.y)

class CircularROI(NamedTuple):
    center: Point
    radius: float
    
    def contains_point(self, point: Point) -> bool:
        distance = math.sqrt((point.x - self.center.x)**2 + (point.y - self.center.y)**2)
        return distance <= self.radius
    
    def get_center(self) -> Point:
        return self.center
    
    def get_area(self) -> float:
        return math.pi * self.radius**2
    
from pydantic import BaseModel, Field
from typing import Union, Literal


# DISK-BASED GLOBAL CONTEXT CACHING

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

# RAG - VECTOR DATABASE MODULE (FAISS)
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
                with open(self.meta_file, 'rb') as f: self.metadata = pickle.load(f)
                logger.info(f"Successfully loaded {self.index.ntotal} vectors.")
            else:
                logger.warning("No existing vector database found. Initializing a new one.")
                self.index = faiss.IndexFlatL2(self.dimension)
        except Exception as e:
            logger.error(f"Failed to load vector database, initializing new one. Error: {e}")
            self.index = faiss.IndexFlatL2(self.dimension)
    
    def save_database(self):
        try:
            logger.info(f"Saving vector database to {self.db_path}...")
            faiss.write_index(self.index, self.index_file)
            with open(self.meta_file, 'wb') as f: pickle.dump(self.metadata, f)
            logger.info("Vector database saved successfully.")
        except Exception as e: logger.error(f"Failed to save vector database: {e}")
    
    def add_entry(self, vector: np.ndarray, metadata_dict: Dict):
        if self.index is None: self.load_database()
        vector = vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(vector)
        self.index.add(vector)
        self.metadata.append(metadata_dict)
        self.save_database()
        logger.info(f"Added new entry. Total entries: {self.index.ntotal}")
    
    def search(self, query_vector: np.ndarray, k: int = 3) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0: return []
        query_vector = query_vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        return [{"metadata": self.metadata[i], "similarity_score": 1 - dist} for i, dist in zip(indices[0], distances[0]) if i != -1]

# LOCAL AI ENGINES

class MedSAM_Lite(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder, self.mask_decoder, self.prompt_encoder = image_encoder, mask_decoder, prompt_encoder
    
    def forward(self, image, points, boxes, masks, tokens=None):
        image_embedding, fs = self.image_encoder(image)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=boxes, masks=masks, tokens=tokens)
        low_res_masks, _ = self.mask_decoder(fs, image_embeddings=image_embedding, image_pe=self.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=False)
        return low_res_masks

class LocalMedSAM:
    def __init__(self, model_path="ai_models/Swin_medsam/model.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.is_ready = None, False
        self.model_path = model_path
    
    def load_model(self):
        try:
            logger.info(f"🔄 Starting MedSAM model loading from {self.model_path} onto {self.device}...")
            
            logger.info("📋 Initializing SwinTransformer image encoder...")
            image_encoder = SwinTransformer()
            logger.info("✅ SwinTransformer image encoder initialized")
            
            logger.info("📋 Initializing PromptEncoder...")
            prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64), input_image_size=(256, 256), mask_in_chans=16)
            logger.info("✅ PromptEncoder initialized")
            
            logger.info("📋 Initializing MaskDecoder with TwoWayTransformer...")
            mask_decoder = MaskDecoder_Prompt(num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8), transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256)
            logger.info("✅ MaskDecoder initialized")
            
            logger.info("📋 Assembling MedSAM_Lite model...")
            self.model = MedSAM_Lite(image_encoder, mask_decoder, prompt_encoder)
            logger.info("✅ MedSAM_Lite model assembled")
            
            logger.info(f"📥 Loading checkpoint from {self.model_path}...")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            logger.info("✅ Checkpoint loaded from disk")
            
            logger.info("📋 Loading state dict into model...")
            self.model.load_state_dict(checkpoint.get('model', checkpoint))
            logger.info("✅ State dict loaded successfully")
            
            logger.info(f"📋 Moving model to {self.device} and setting eval mode...")
            self.model.to(self.device).eval()
            self.is_ready = True
            logger.info("🎉 MedSAM model loaded successfully and ready for inference!")
            
        except Exception as e: 
            logger.error(f"❌ Failed to load MedSAM model: {e}")
            import traceback
            logger.error(f"📋 Full error traceback: {traceback.format_exc()}")
    
    @torch.no_grad()
    def get_mask(self, image_np, point_prompt):
        if not self.is_ready: return None
        h, w = image_np.shape[:2]
        norm_img = (image_np - image_np.min()) / np.ptp(image_np) * 255 if np.ptp(image_np) > 0 else np.zeros_like(image_np)
        img_3c = np.repeat(norm_img[:, :, None], 3, -1) if norm_img.ndim == 2 else norm_img
        scale = 256 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img_3c, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_padded = np.pad(img_resized, ((0, 256 - new_h), (0, 256 - new_w), (0, 0)))
        img_tensor = torch.as_tensor(img_padded).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        points = torch.as_tensor([[[point_prompt[0] * scale, point_prompt[1] * scale]]], dtype=torch.float, device=self.device)
        labels = torch.as_tensor([[1]], dtype=torch.int, device=self.device)
        masks = self.model(img_tensor, (points, labels), None, torch.zeros((1, 1, 256, 256), dtype=torch.float).to(self.device))
        masks = F.interpolate(F.interpolate(masks, (256, 256), mode="bilinear", align_corners=False)[:, :, :new_h, :new_w], (h, w), mode="bilinear", align_corners=False)
        return (torch.sigmoid(masks[0,0]) > 0.5).cpu().numpy().astype(np.uint8)

class LocalROIAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor, self.is_ready = None, None, False
        self.candidate_labels = ["normal soft tissue", "fat", "fluid or cyst", "bone or calcification", "air or gas", "a high-density lesion", "a low-density lesion", "a blood vessel", "lung tissue"]
        self.text_prompts = [f"a medical image of {label}" for label in self.candidate_labels]
    
    def _monitor_download_progress(self, cache_dir="/root/.cache/huggingface", expected_size_mb=3354, stop_event=None):
        """Monitor HuggingFace model download progress"""
        start_time = time.time()
        prev_size = 0

        while not stop_event.is_set():
            try:
                # Get current cache size
                result = os.popen(f"du -sm {cache_dir} 2>/dev/null || echo '0'").read().strip()
                current_size = int(result.split()[0]) if result and result.split()[0].isdigit() else 0

                # Calculate progress
                progress = min(current_size / expected_size_mb * 100, 100)
                speed = (current_size - prev_size) / 2 if prev_size > 0 else 0  # MB/s (2 second intervals)
                elapsed = time.time() - start_time

                # Create progress bar
                bar_length = 30
                filled_length = int(bar_length * progress / 100)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)

                # ETA calculation
                if speed > 0:
                    remaining_mb = expected_size_mb - current_size
                    eta_seconds = remaining_mb / speed
                    eta_str = f"ETA: {int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"
                else:
                    eta_str = "ETA: --:--"

                logger.info(f"📥 [{bar}] {progress:.1f}% ({current_size}/{expected_size_mb}MB) {speed:.1f}MB/s {eta_str}")
                prev_size = current_size

                # Break if download complete
                if current_size >= expected_size_mb * 0.95:  # 95% threshold to account for file overhead
                    break

                time.sleep(2)
            except Exception as e:
                logger.warning(f"Progress monitoring error: {e}")
                time.sleep(2)

    def load_model(self):
        try:
            logger.info(f"🔄 Starting MedSigLIP model loading onto {self.device}...")
            model_name = "google/medsiglip-448"

            # Check if model is already cached
            cache_dir = "/root/.cache/huggingface"
            cache_size = 0
            try:
                result = os.popen(f"du -sm {cache_dir} 2>/dev/null || echo '0'").read().strip()
                cache_size = int(result.split()[0]) if result and result.split()[0].isdigit() else 0
            except:
                pass

            if cache_size < 3000:  # If less than 3GB cached, show progress
                logger.info(f"📥 Downloading MedSigLIP model: {model_name} (3.35GB)")
                logger.info("🔄 Starting download progress monitor...")

                # Start progress monitoring in background thread
                stop_event = threading.Event()
                progress_thread = threading.Thread(
                    target=self._monitor_download_progress,
                    args=(cache_dir, 3354, stop_event)
                )
                progress_thread.daemon = True
                progress_thread.start()

                # Load model (this will trigger download with progress monitoring)
                self.model = AutoModel.from_pretrained(model_name)

                # Stop progress monitoring
                stop_event.set()
                progress_thread.join(timeout=1)
                logger.info("✅ MedSigLIP model downloaded and loaded from HuggingFace")
            else:
                logger.info(f"📋 Loading cached MedSigLIP model: {model_name}")
                self.model = AutoModel.from_pretrained(model_name)
                logger.info("✅ MedSigLIP model loaded from cache (instant)")

            logger.info(f"📋 Moving MedSigLIP model to {self.device}...")
            self.model = self.model.to(self.device)
            logger.info(f"✅ MedSigLIP model moved to {self.device}")

            logger.info(f"📥 Loading MedSigLIP processor: {model_name}...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            logger.info("✅ MedSigLIP processor loaded")

            self.is_ready = True
            logger.info("🎉 Local ROI Analyzer is ready for inference!")

        except Exception as e:
            logger.error(f"❌ Failed to load MedSigLIP model: {e}")
            import traceback
            logger.error(f"📋 Full error traceback: {traceback.format_exc()}")
    
    @torch.no_grad()
    def analyze_roi(self, roi_image):
        if not self.is_ready: return {"error": "Local AI model not ready."}
        try:
            inputs = self.processor(text=self.text_prompts, images=roi_image, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            logits, image_embeds = outputs.logits_per_image, outputs.image_embeds.cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            analysis = {"label": self.candidate_labels[np.argmax(probs)]}
            return {"analysis": analysis, "embedding": image_embeds}
        except Exception as e: return {"error": f"Local analysis failed: {e}"}

# CACHE AND UTILITIES

class BackendCache:
    def __init__(self):
        self.current_volume, self.current_slices, self.current_series_info = None, None, {}
        self.is_processing, self.progress_message = False, "Idle"
    
    def set_progress(self, msg: str):
        self.progress_message = msg
        logger.info(f"Progress: {msg}")
    
    def clear_all_context(self):
        self.current_volume, self.current_slices = None, None
        gc.collect()

# =============================================================================
# ANATOMICAL ANALYSIS FUNCTIONS
# =============================================================================
def analyze_roi_anatomical_overlap(roi_region: Union[RectangularROI, CircularROI], anatomical_map: Dict) -> Dict:
    """
    Analyze what anatomical structures overlap with the ROI region.
    Returns detailed analysis of overlapping structures.
    """
    if not anatomical_map or "structures" not in anatomical_map:
        return {"overlapping_structures": [], "primary_anatomy": "uncharacterized region", "coverage_analysis": {}}
    
    overlapping_structures = []
    total_roi_area = roi_region.get_area()
    
    for structure in anatomical_map["structures"]:
        bbox = structure.get("bounding_box", {})
        if not all(key in bbox for key in ["x_min", "y_min", "x_max", "y_max"]):
            continue
        
        # Convert normalized coordinates to canvas coordinates
        struct_bbox = RectangularROI(
            Point(bbox["x_min"] * COORDINATE_SCALE, bbox["y_min"] * COORDINATE_SCALE),
            Point(bbox["x_max"] * COORDINATE_SCALE, bbox["y_max"] * COORDINATE_SCALE)
        )
        
        # Calculate overlap
        overlap_info = calculate_overlap(roi_region, struct_bbox)
        
        if overlap_info["has_overlap"]:
            overlap_data = {
                "structure": structure,
                "overlap_area": overlap_info["overlap_area"],
                "overlap_percentage": overlap_info["overlap_percentage"],
                "roi_coverage": overlap_info["overlap_area"] / total_roi_area * 100,
                "structure_coverage": overlap_info["overlap_area"] / struct_bbox.get_area() * 100,
                "center_distance": calculate_distance(roi_region.get_center(), struct_bbox.get_center())
            }
            overlapping_structures.append(overlap_data)
    
    # Sort by overlap percentage (highest first)
    overlapping_structures.sort(key=lambda x: x["overlap_percentage"], reverse=True)
    
    # Determine primary anatomy
    primary_anatomy = "uncharacterized region"
    if overlapping_structures:
        # Use the structure with highest ROI coverage as primary
        primary_structure = max(overlapping_structures, key=lambda x: x["roi_coverage"])
        primary_anatomy = primary_structure["structure"]["label"]
    
    coverage_analysis = {
        "total_structures_found": len(overlapping_structures),
        "roi_area": total_roi_area,
        "primary_structure_coverage": overlapping_structures[0]["roi_coverage"] if overlapping_structures else 0,
        "multi_structure_roi": len(overlapping_structures) > 1
    }
    
    return {
        "overlapping_structures": overlapping_structures,
        "primary_anatomy": primary_anatomy,
        "coverage_analysis": coverage_analysis
    }

def calculate_overlap(roi_region: Union[RectangularROI, CircularROI], 
                     structure_bbox: RectangularROI) -> Dict:
    """Calculate overlap between ROI region and anatomical structure bounding box."""
    
    if isinstance(roi_region, RectangularROI):
        return calculate_rectangle_overlap(roi_region, structure_bbox)
    elif isinstance(roi_region, CircularROI):
        return calculate_circle_rectangle_overlap(roi_region, structure_bbox)
    else:
        return {"has_overlap": False, "overlap_area": 0, "overlap_percentage": 0}

def calculate_rectangle_overlap(roi_rect: RectangularROI, struct_rect: RectangularROI) -> Dict:
    """Calculate overlap between two rectangles."""
    # Find intersection rectangle
    left = max(roi_rect.top_left.x, struct_rect.top_left.x)
    top = max(roi_rect.top_left.y, struct_rect.top_left.y)
    right = min(roi_rect.bottom_right.x, struct_rect.bottom_right.x)
    bottom = min(roi_rect.bottom_right.y, struct_rect.bottom_right.y)
    
    if left < right and top < bottom:
        overlap_area = (right - left) * (bottom - top)
        struct_area = struct_rect.get_area()
        overlap_percentage = (overlap_area / struct_area * 100) if struct_area > 0 else 0
        
        return {
            "has_overlap": True,
            "overlap_area": overlap_area,
            "overlap_percentage": overlap_percentage
        }
    
    return {"has_overlap": False, "overlap_area": 0, "overlap_percentage": 0}

def calculate_circle_rectangle_overlap(roi_circle: CircularROI, struct_rect: RectangularROI) -> Dict:
    """Approximate overlap between circle and rectangle."""
    # Simplified approach: check if circle center is in rectangle or if rectangle corners are in circle
    center_in_rect = struct_rect.contains_point(roi_circle.center)
    
    # Check rectangle corners
    corners = [
        struct_rect.top_left,
        Point(struct_rect.bottom_right.x, struct_rect.top_left.y),
        struct_rect.bottom_right,
        Point(struct_rect.top_left.x, struct_rect.bottom_right.y)
    ]
    
    corners_in_circle = sum(1 for corner in corners if roi_circle.contains_point(corner))
    
    if center_in_rect or corners_in_circle > 0:
        # Rough approximation - in practice, you might want more precise geometric calculation
        circle_area = roi_circle.get_area()
        rect_area = struct_rect.get_area()
        
        if center_in_rect:
            # Circle center is in rectangle - assume significant overlap
            estimated_overlap = min(circle_area, rect_area) * 0.7  # Conservative estimate
        else:
            # Some corners in circle - partial overlap
            estimated_overlap = min(circle_area, rect_area) * (corners_in_circle / 4) * 0.5
        
        overlap_percentage = (estimated_overlap / rect_area * 100) if rect_area > 0 else 0
        
        return {
            "has_overlap": True,
            "overlap_area": estimated_overlap,
            "overlap_percentage": overlap_percentage
        }
    
    return {"has_overlap": False, "overlap_area": 0, "overlap_percentage": 0}

def calculate_distance(point1: Point, point2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def find_overlapping_structures(canvas_x: float, canvas_y: float, anatomical_map: Dict) -> List[Dict]:
    """
    Find anatomical structures that overlap with the given canvas coordinates.
    (Legacy function for backward compatibility)
    """
    # Convert canvas coordinates (0-COORDINATE_SCALE) to normalized coordinates (0-1)
    norm_x = canvas_x / float(COORDINATE_SCALE)
    norm_y = canvas_y / float(COORDINATE_SCALE)
    
    overlapping_structures = []
    
    # Check if anatomical_map has structures
    if not anatomical_map or "structures" not in anatomical_map:
        return overlapping_structures
    
    for structure in anatomical_map["structures"]:
        bbox = structure.get("bounding_box", {})
        
        # Check if the point is within the bounding box
        if (bbox.get("x_min", 0) <= norm_x <= bbox.get("x_max", 1) and 
            bbox.get("y_min", 0) <= norm_y <= bbox.get("y_max", 1)):
            
            # Calculate overlap area or confidence score
            overlap_info = {
                "structure": structure,
                "distance_from_center": calculate_distance_from_bbox_center(norm_x, norm_y, bbox),
                "bbox_area": (bbox.get("x_max", 1) - bbox.get("x_min", 0)) * (bbox.get("y_max", 1) - bbox.get("y_min", 0))
            }
            overlapping_structures.append(overlap_info)
    
    # Sort by distance from center (closest first) and then by smaller area (more specific structures first)
    overlapping_structures.sort(key=lambda x: (x["distance_from_center"], x["bbox_area"]))
    
    return overlapping_structures

def calculate_distance_from_bbox_center(x: float, y: float, bbox: Dict) -> float:
    """
    Calculate the distance from a point to the center of a bounding box.
    """
    center_x = (bbox.get("x_min", 0) + bbox.get("x_max", 1)) / 2
    center_y = (bbox.get("y_min", 0) + bbox.get("y_max", 1)) / 2
    
    return math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

# =============================================================================
# GOOGLE CLOUD UTILITIES
# =============================================================================
def get_gcloud_access_token():
    try:
        logger.info("Attempting to get GCP credentials...")
        creds, project = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        logger.info(f"Found credentials for project: {project}")
        
        logger.info("Refreshing credentials...")
        creds.refresh(google.auth.transport.requests.Request())
        
        if not creds.token:
            logger.error("No token received after refresh")
            return None
            
        logger.info(f"Successfully obtained access token (expires: {creds.expiry})")
        return creds.token
        
    except google.auth.exceptions.DefaultCredentialsError as cred_err:
        logger.error(f"GCP credentials error: {cred_err}")
        logger.error("Make sure you've run 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error getting GCP token: {e}")
        return None

def check_and_upload_slice_to_gcs(slice_2d, slice_id, series_uid):
    proj_id, bkt_name = os.getenv("PROJECT_ID"), os.getenv("GCS_BUCKET_NAME")
    if not all([proj_id, bkt_name]): 
        logger.error("GCS config missing.")
        return None
    try:
        client = storage.Client(project=proj_id)
        bucket = client.bucket(bkt_name)
        norm = (slice_2d - np.min(slice_2d)) / np.ptp(slice_2d) * 255 if np.ptp(slice_2d) > 0 else np.zeros_like(slice_2d)
        img_hash = hashlib.md5(norm.tobytes()).hexdigest()
        blob_path = f"uploaded_images/slice_{series_uid}_{slice_id}_{img_hash}.png"
        blob = bucket.blob(blob_path)
        if not blob.exists():
            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                Image.fromarray(norm.astype(np.uint8), 'L').convert('RGB').save(tmp.name, 'PNG')
                blob.upload_from_filename(tmp.name)
        return blob.generate_signed_url(version="v4", expiration=timedelta(minutes=15), method="GET")
    except Exception as e: 
        logger.error(f"GCS Error: {e}")
        return None

async def call_medgamma_inference(signed_url, prompt):
    proj_id, ep_id, ep_url = os.getenv("PROJECT_ID"), os.getenv("ENDPOINT_ID"), os.getenv("ENDPOINT_URL")
    
    # Enhanced debugging for configuration
    logger.info(f"MedGamma API Config Check:")
    logger.info(f"  PROJECT_ID: {'✓' if proj_id else '✗'} ({proj_id[:10]}... if provided)")
    logger.info(f"  ENDPOINT_ID: {'✓' if ep_id else '✗'} ({ep_id[:10]}... if provided)")
    logger.info(f"  ENDPOINT_URL: {'✓' if ep_url else '✗'} ({ep_url[:30]}... if provided)")
    
    if not all([proj_id, ep_id, ep_url]): 
        logger.error("Missing MedGamma endpoint configuration")
        return {"error": "Endpoint config missing."}
    
    token = get_gcloud_access_token()
    if not token: 
        logger.error("Failed to get GCP access token")
        return {"error": "GCP auth failed."}
    
    logger.info(f"GCP token obtained: {token[:20]}...")
    
    payload = {
        "instances": [{
            "@requestFormat": "chatCompletions",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a world-class radiologist with expertise in comprehensive anatomical analysis and pathology detection."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": signed_url}}
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40
        }]
    }
    
    #url = f"https://{ep_url.replace('https://', '')}/v1/projects/{proj_id}/locations/us-central1/endpoints/{ep_id}:predict"
    url = f"{ep_url}/v1/projects/{proj_id}/locations/us-central1/endpoints/{ep_id}:predict"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    logger.info(f"Making request to: {url}")
    logger.info(f"Payload structure: instances[0] has {len(payload['instances'][0]['messages'])} messages")
    logger.info(f"Image URL accessible: {signed_url[:50]}...")
    
    try:
        loop = asyncio.get_event_loop()
        
        # Add more detailed logging
        logger.info("Sending request to MedGamma API...")
        start_time = time.time()
        
        response = await loop.run_in_executor(
            None, 
            lambda: requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=400
            )
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"API response received in {elapsed_time:.2f} seconds")
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        
        # Log response content for debugging (first 500 chars)
        response_text = response.text
        logger.info(f"Response content preview: {response_text[:500]}...")
        
        response.raise_for_status()
        
        try:
            response_json = response.json()
            logger.info("Successfully parsed JSON response")
            return response_json
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse JSON response: {json_err}")
            logger.error(f"Raw response: {response_text}")
            return {"error": "Invalid JSON response from API", "details": response_text[:1000]}
        
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"API request timed out after 300 seconds (5 minutes): {timeout_err}")
        return {"error": "API request timed out", "details": str(timeout_err)}
        
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error: {conn_err}")
        return {"error": "Connection failed", "details": str(conn_err)}
        
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error: {http_err}")
        error_details = ""
        if hasattr(http_err, 'response') and http_err.response is not None:
            error_details = http_err.response.text
            logger.error(f"Error response body: {error_details}")
        return {"error": "HTTP error", "details": error_details or str(http_err)}
        
    except requests.exceptions.RequestException as req_err:
        logger.error(f"General request error: {req_err}")
        error_details = ""
        if hasattr(req_err, 'response') and req_err.response is not None:
            error_details = req_err.response.text
            logger.error(f"Error response body: {error_details}")
        else:
            logger.error("No response object available")
        return {"error": "API request failed", "details": error_details or str(req_err)}
        
    except Exception as unexpected_err:
        logger.error(f"Unexpected error: {unexpected_err}")
        return {"error": "Unexpected error", "details": str(unexpected_err)}

# =============================================================================
# DICOM UTILITIES
# =============================================================================
class SeriesInfo(BaseModel):
    uid: str
    description: str
    slices: int
    patient_id: str
    modality: str
    display_name: str

class LoadSeriesResponse(BaseModel):
    status: str
    shape: List[int]
    value_range: List[float]
    metadata: Dict[str, Any]

def scan_for_series(dir_path):
    """
    Scan for DICOM series in a directory. Handles both absolute and relative paths.
    For security, paths are resolved relative to DICOM_DATA_ROOT in Docker environments.
    """
    # Resolve the path safely
    resolved_path = resolve_safe_path(dir_path)

    logger.info(f"Scanning for DICOM series in: {resolved_path}")

    if not os.path.exists(resolved_path):
        logger.warning(f"Directory does not exist: {resolved_path}")
        return {}

    if not os.path.isdir(resolved_path):
        logger.warning(f"Path is not a directory: {resolved_path}")
        return {}

    s_map = defaultdict(list)
    file_count = 0

    for root, _, files in os.walk(resolved_path):
        for f in files:
            # Only process files that look like DICOM files
            if f.lower().endswith(('.dcm', '.dicom', '.ima')) or '.' not in f:
                file_path = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    if hasattr(ds, 'SeriesInstanceUID') and ds.SeriesInstanceUID:
                        s_map[ds.SeriesInstanceUID].append(file_path)
                        file_count += 1
                except Exception as e:
                    # Skip non-DICOM files silently
                    continue

    logger.info(f"Found {file_count} DICOM files across {len(s_map)} series")

    s_info = {}
    for uid, files in s_map.items():
        try:
            # Sort files for consistent ordering
            files.sort()
            ds = pydicom.dcmread(files[0])
            s_info[uid] = {
                "uid": uid,
                "description": getattr(ds, "SeriesDescription", "N/A"),
                "slices": len(files),
                "patient_id": getattr(ds, 'PatientID', 'N/A'),
                "modality": getattr(ds, 'Modality', 'N/A'),
                "files": files
            }
        except Exception as e:
            logger.warning(f"Error processing series {uid}: {e}")
            continue

    return s_info

def resolve_safe_path(user_path):
    """
    Safely resolve a user-provided path relative to DICOM_DATA_ROOT.
    Prevents directory traversal attacks and ensures paths stay within mounted volumes.
    All paths are treated as relative to the mounted DICOM data directory.
    """
    # Handle empty or root path
    if not user_path or user_path in ['/', '']:
        return DICOM_DATA_ROOT

    # Remove leading slashes to make all paths relative
    user_path = user_path.lstrip('/')

    # Join with DICOM_DATA_ROOT
    resolved_path = os.path.join(DICOM_DATA_ROOT, user_path)

    # Normalize path to handle .. and . components
    resolved_path = os.path.normpath(resolved_path)

    # Security check: ensure path is within DICOM_DATA_ROOT
    normalized_root = os.path.normpath(DICOM_DATA_ROOT)
    if not resolved_path.startswith(normalized_root):
        logger.warning(f"Path traversal attempt detected: {user_path} -> {resolved_path}")
        resolved_path = DICOM_DATA_ROOT

    return resolved_path

def load_dicom_volume(files):
    slices = [pydicom.dcmread(f) for f in files if hasattr(pydicom.dcmread(f), 'pixel_array')]
    if not slices: return None, None
    try:
        if hasattr(slices[0], 'ImagePositionPatient'): slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
        elif hasattr(slices[0], 'InstanceNumber'): slices.sort(key=lambda s: int(s.InstanceNumber))
    except (AttributeError, TypeError): logger.warning("Could not sort slices.")
    vol = np.stack([s.pixel_array.astype(np.float64) for s in slices])
    if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
        vol = vol * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)
    return slices, vol

# =============================================================================
# 3D MESH GENERATION PIPELINE
# =============================================================================

class MeshGenerationStatus:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()

    def set_status(self, job_id: str, status: str, progress: float = 0, message: str = "", mesh_url: str = None, error: str = None):
        with self.lock:
            self.jobs[job_id] = {
                "status": status,
                "progress": progress,
                "message": message,
                "mesh_url": mesh_url,
                "error": error,
                "timestamp": datetime.now().isoformat()
            }

    def get_status(self, job_id: str):
        with self.lock:
            return self.jobs.get(job_id)

def preprocess_volume_for_3d(volume: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """
    Preprocess DICOM volume for 3D mesh generation using windowing.

    Args:
        volume: 3D numpy array of DICOM data
        window_center: Center of the intensity window (Hounsfield Units)
        window_width: Width of the intensity window (Hounsfield Units)

    Returns:
        Normalized volume array (0-1 range) ready for marching cubes
    """
    volume = volume.astype(np.float32)

    # Apply windowing
    low = window_center - window_width / 2
    high = window_center + window_width / 2
    volume = np.clip(volume, low, high)

    # Normalize to 0-1 range
    volume -= volume.min()
    max_val = volume.max()
    if max_val > 0:
        volume /= max_val

    return volume

def create_3d_mesh(volume: np.ndarray, spacing: tuple, threshold: float, decimation_factor: float = 0.9) -> dict:
    """
    Generate 3D mesh from volume using marching cubes algorithm.

    Args:
        volume: Preprocessed 3D volume (0-1 normalized)
        spacing: Voxel spacing (z, y, x) in mm
        threshold: Isosurface threshold (0-1 range)
        decimation_factor: Reduction factor for mesh simplification (0-1)

    Returns:
        Dictionary containing vertices, faces, and metadata
    """
    # Import trimesh locally to avoid scoping issues
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh library is not available. Install it with: pip install trimesh")

    try:
        # Apply marching cubes algorithm
        verts, faces, normals, values = measure.marching_cubes(
            volume,
            level=threshold,
            spacing=spacing,
            gradient_direction='descent'
        )

        # Create trimesh object for processing
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

        # Apply mesh processing
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()

        # Smooth the mesh using Laplacian smoothing
        try:
            mesh = mesh.smoothed()
        except AttributeError:
            # trimesh doesn't have smoothed() method, use alternative approach
            try:
                mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=1)
            except (ImportError, AttributeError):
                logger.warning("trimesh.smoothing not available, skipping mesh smoothing")
                pass

        # Decimate mesh to reduce file size
        if 0 < decimation_factor < 1:
            target_faces = int(len(mesh.faces) * (1 - decimation_factor))
            if target_faces > 100:  # Ensure minimum face count
                try:
                    mesh = mesh.simplify_quadric_decimation(target_faces)
                except ImportError as e:
                    logger.warning(f"Mesh decimation not available (missing fast_simplification): {e}")
                    # Continue without decimation
                except Exception as e:
                    logger.warning(f"Mesh decimation failed: {e}")
                    # Continue without decimation

        # Center the mesh
        mesh.vertices -= mesh.center_mass

        return {
            "vertices": mesh.vertices.tolist(),
            "faces": mesh.faces.tolist(),
            "normals": mesh.vertex_normals.tolist(),
            "face_count": len(mesh.faces),
            "vertex_count": len(mesh.vertices),
            "volume": mesh.volume,
            "center_mass": mesh.center_mass.tolist(),
            "bounds": mesh.bounds.tolist()
        }

    except Exception as e:
        logger.error(f"Error in mesh generation: {str(e)}")
        raise ValueError(f"Mesh generation failed: {str(e)}")

def export_mesh_to_glb(mesh_data: dict, output_path: str) -> str:
    """
    Export mesh data to GLB format for web viewing.

    Args:
        mesh_data: Dictionary containing vertices, faces, and normals
        output_path: Full path where GLB file should be saved

    Returns:
        Path to the saved GLB file
    """
    # Import trimesh locally to avoid scoping issues
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh library is not available. Install it with: pip install trimesh")

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create trimesh object from data
        mesh = trimesh.Trimesh(
            vertices=np.array(mesh_data["vertices"]),
            faces=np.array(mesh_data["faces"]),
            vertex_normals=np.array(mesh_data["normals"])
        )

        # Apply material for better visualization
        try:
            # Create a material if it doesn't exist
            if not hasattr(mesh.visual, 'material') or mesh.visual.material is None:
                mesh.visual.material = trimesh.visual.material.PBRMaterial()

            mesh.visual.material.diffuse = [200, 200, 200, 255]  # Light gray
            mesh.visual.material.metallic = 0.1
            mesh.visual.material.roughness = 0.8
        except (AttributeError, TypeError) as e:
            logger.warning(f"Could not set material properties: {e}")
            # Continue without material properties

        # Export as GLB
        with open(output_path, 'wb') as f:
            f.write(trimesh.exchange.gltf.export_glb(mesh))

        logger.info(f"Successfully exported mesh to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error exporting mesh to GLB: {str(e)}")
        raise ValueError(f"GLB export failed: {str(e)}")

def process_dicom_to_3d_mesh(series_uid: str, threshold_hu: int, decimation_factor: float, job_id: str, status_manager: MeshGenerationStatus):
    """
    Background task to process DICOM series into 3D mesh.

    Args:
        series_uid: Series UID to process
        threshold_hu: Threshold in Hounsfield Units
        decimation_factor: Mesh reduction factor (0-1)
        job_id: Unique job identifier
        status_manager: Status tracking manager
    """
    try:
        status_manager.set_status(job_id, "processing", 0.1, "Loading DICOM series...")

        # Validate series exists
        if series_uid not in backend_cache.current_series_info:
            raise ValueError(f"Series {series_uid} not found in current session")

        # Load DICOM volume if not already loaded
        if backend_cache.current_volume is None or list(backend_cache.current_series_info.keys())[0] != series_uid:
            status_manager.set_status(job_id, "processing", 0.2, "Loading DICOM volume...")
            slices, volume = load_dicom_volume(backend_cache.current_series_info[series_uid]["files"])
            if volume is None:
                raise ValueError("Failed to load DICOM volume")
            backend_cache.current_volume = volume
            backend_cache.current_slices = slices

        volume = backend_cache.current_volume
        slices = backend_cache.current_slices

        # Get spacing information
        if hasattr(slices[0], 'PixelSpacing') and hasattr(slices[0], 'SliceThickness'):
            pixel_spacing = list(slices[0].PixelSpacing)
            slice_thickness = float(slices[0].SliceThickness)
            spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])
        else:
            # Default spacing if not available
            spacing = (1.0, 1.0, 1.0)
            logger.warning("Using default spacing (1,1,1) - actual spacing not found in DICOM headers")

        status_manager.set_status(job_id, "processing", 0.4, "Preprocessing volume for 3D generation...")

        # Preprocess volume for bone visualization
        bone_window_center = 300
        bone_window_width = 1500
        processed_volume = preprocess_volume_for_3d(volume, bone_window_center, bone_window_width)

        # Convert HU threshold to normalized level
        normalized_threshold = (threshold_hu - (bone_window_center - bone_window_width / 2)) / bone_window_width
        normalized_threshold = max(0.01, min(normalized_threshold, 0.99))  # Clamp to safe range

        status_manager.set_status(job_id, "processing", 0.6, "Generating 3D mesh using marching cubes...")

        # Generate mesh
        mesh_data = create_3d_mesh(processed_volume, spacing, normalized_threshold, decimation_factor)

        status_manager.set_status(job_id, "processing", 0.8, "Exporting mesh to GLB format...")

        # Create output directory
        mesh_dir = "static/meshes"
        os.makedirs(mesh_dir, exist_ok=True)

        # Export mesh
        glb_path = os.path.join(mesh_dir, f"{job_id}.glb")
        export_mesh_to_glb(mesh_data, glb_path)

        # Generate mesh URL
        mesh_url = f"/api/meshes/{job_id}.glb"

        status_manager.set_status(
            job_id,
            "completed",
            1.0,
            f"3D mesh generated successfully. Faces: {mesh_data['face_count']}, Vertices: {mesh_data['vertex_count']}",
            mesh_url=mesh_url
        )

        # Track metrics
        with metrics_lock:
            disease_metrics["model_predictions"] += 1
            disease_metrics["successful_requests"] += 1

        logger.info(f"Successfully generated 3D mesh for job {job_id}")

    except Exception as e:
        error_msg = f"3D mesh generation failed: {str(e)}"
        logger.error(error_msg)
        status_manager.set_status(job_id, "failed", 0, "", error=error_msg)

        # Track metrics
        with metrics_lock:
            disease_metrics["failed_requests"] += 1

# =============================================================================
# INITIALIZE COMPONENTS
# =============================================================================
local_medsam = LocalMedSAM()
local_analyzer = LocalROIAnalyzer()
vector_db_manager = VectorDBManager()
backend_cache = BackendCache()
disk_global_cache = DiskBasedGlobalContextCache()
mesh_status_manager = MeshGenerationStatus()

# =============================================================================
# CORE API ENDPOINTS
# =============================================================================
@app.on_event("startup")
async def startup_event():
    logger.info("✅ Spatial-Priority Medical RAG System starting up...")
    logger.info("🔧 Loading AI models and initializing components...")

    try:
        logger.info("📋 Loading MedSigLIP ROI analyzer...")
        local_analyzer.load_model()
        logger.info("✅ MedSigLIP ROI analyzer loaded successfully")

        logger.info("📋 Loading MedSAM segmentation model...")
        local_medsam.load_model()
        logger.info("✅ MedSAM segmentation model loaded successfully")

        logger.info("📋 Loading vector database...")
        vector_db_manager.load_database()
        logger.info("✅ Vector database loaded successfully")

        logger.info("🎉 All components initialized successfully!")
        logger.info("🌐 Backend is ready to accept requests")

    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {str(e)}")
        logger.error("⚠️  Some features may not work properly")
        import traceback
        logger.error(f"📋 Full error traceback: {traceback.format_exc()}")

@app.get("/")
async def root(): 
    return {"message": "Spatial-Priority Medical RAG System is running.", "version": app.version}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy" if all([local_medsam.is_ready, local_analyzer.is_ready]) else "degraded",
        "models": {
            "medsam_ready": local_medsam.is_ready,
            "analyzer_ready": local_analyzer.is_ready
        },
        "cache": {
            "series_loaded": backend_cache.current_volume is not None,
            "global_cache_entries": len(disk_global_cache.memory_cache)
        }
    }

@app.get("/api/metrics")
async def get_rag_metrics():
    """Get RAG system metrics for monitoring dashboard"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        with metrics_lock:
            # Calculate average response time
            avg_response_time = sum(disease_metrics["response_times"]) / len(disease_metrics["response_times"]) if disease_metrics["response_times"] else 0

            # Calculate uptime
            uptime_seconds = (datetime.now() - disease_metrics["service_start_time"]).total_seconds()

            # Calculate success rate
            success_rate = (disease_metrics["successful_requests"] / disease_metrics["total_requests"] * 100) if disease_metrics["total_requests"] > 0 else 0

            current_metrics = {
                "service": "rag",
                "status": "healthy" if all([local_medsam.is_ready, local_analyzer.is_ready]) else "degraded",
                "uptime_seconds": uptime_seconds,
                "uptime_formatted": f"{uptime_seconds // 3600:.0f}h {(uptime_seconds % 3600) // 60:.0f}m",
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_total_gb": round(memory.total / (1024**3), 2),
                    "memory_used_gb": round(memory.used / (1024**3), 2),
                    "memory_percent": memory.percent,
                    "disk_total_gb": round(disk.total / (1024**3), 2),
                    "disk_used_gb": round(disk.used / (1024**3), 2),
                    "disk_percent": round(disk.used / disk.total * 100, 1)
                },
                "requests": {
                    "total": disease_metrics["total_requests"],
                    "successful": disease_metrics["successful_requests"],
                    "failed": disease_metrics["failed_requests"],
                    "model_predictions": disease_metrics["model_predictions"],
                    "dicom_files_processed": disease_metrics["dicom_files_processed"],
                    "cache_operations": disease_metrics["cache_operations"],
                    "roi_analyses": disease_metrics["roi_analyses"],
                    "success_rate": round(success_rate, 2),
                    "active_sessions": disease_metrics["active_sessions"],
                    "last_request": disease_metrics["last_request_time"].isoformat() if disease_metrics["last_request_time"] else None
                },
                "performance": {
                    "avg_response_time_ms": round(avg_response_time, 2),
                    "recent_response_times": disease_metrics["response_times"][-10:] if disease_metrics["response_times"] else [],
                    "memory_peak_usage": disease_metrics["memory_peak_usage"]
                },
                "models": {
                    "medsam_ready": local_medsam.is_ready,
                    "analyzer_ready": local_analyzer.is_ready,
                    "models_loaded": all([local_medsam.is_ready, local_analyzer.is_ready])
                },
                "data": {
                    "series_loaded": backend_cache.current_volume is not None,
                    "global_cache_entries": len(disk_global_cache.memory_cache),
                    "current_series_info": len(backend_cache.current_series_info) if backend_cache.current_series_info else 0
                },
                "timestamp": datetime.now().isoformat()
            }

        return current_metrics

    except Exception as e:
        logger.error(f"Error getting RAG metrics: {e}")
        return {
            "service": "rag",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/scan-directory", response_model=List[SeriesInfo])
async def scan_directory(directory: str = Query(..., description="Directory path to scan (relative to mounted DICOM data)")):
    """
    Scan for DICOM series in the specified directory.
    The directory path should be relative to the mounted DICOM data root.
    Users select this path through the GUI after browsing the available directories.
    """
    start_time = time.time()

    # Track request metrics
    with metrics_lock:
        disease_metrics["total_requests"] += 1
        disease_metrics["last_request_time"] = datetime.now()
        disease_metrics["active_sessions"] += 1

    try:
        logger.info(f"Scanning directory requested: {directory}")
        series_info = scan_for_series(directory)
        backend_cache.current_series_info = series_info

        if not series_info:
            logger.warning(f"No DICOM series found in directory: {directory}")
            result = []
        else:
            result = [SeriesInfo(**info, display_name=f"{info['patient_id']}-{info['modality']}({info['slices']})") for _, info in series_info.items()]
            logger.info(f"Found {len(result)} DICOM series in directory: {directory}")

        # Track success metrics
        response_time = (time.time() - start_time) * 1000
        with metrics_lock:
            disease_metrics["successful_requests"] += 1
            disease_metrics["dicom_files_processed"] += len(result)
            disease_metrics["response_times"].append(response_time)
            disease_metrics["active_sessions"] = max(0, disease_metrics["active_sessions"] - 1)
            # Keep only last 100 response times
            if len(disease_metrics["response_times"]) > 100:
                disease_metrics["response_times"] = disease_metrics["response_times"][-100:]

        return result

    except Exception as e:
        with metrics_lock:
            disease_metrics["failed_requests"] += 1
            disease_metrics["active_sessions"] = max(0, disease_metrics["active_sessions"] - 1)
        logger.error(f"Error scanning directory: {e}")
        raise

@app.get("/api/browse-directories")
async def browse_directories(path: str = Query("/", description="Directory path to browse")):
    """
    Browse directories and files for DICOM directory selection.
    Paths are resolved relative to DICOM_DATA_ROOT for security and portability.
    """
    try:
        # Resolve path safely relative to DICOM_DATA_ROOT
        if path == "/" or path == "":
            # Start from the root of mounted DICOM data
            resolved_path = DICOM_DATA_ROOT
            display_path = "/"
        else:
            resolved_path = resolve_safe_path(path)
            # Calculate display path relative to DICOM_DATA_ROOT
            try:
                display_path = "/" + os.path.relpath(resolved_path, DICOM_DATA_ROOT)
                if display_path == "/.":
                    display_path = "/"
            except ValueError:
                display_path = "/"

        logger.info(f"Browsing directory: {resolved_path} (display: {display_path})")

        if not os.path.exists(resolved_path):
            raise HTTPException(status_code=404, detail=f"Path does not exist: {display_path}")

        if not os.path.isdir(resolved_path):
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {display_path}")

        items = []

        # Add parent directory if not at root
        if resolved_path != DICOM_DATA_ROOT:
            parent_path = os.path.dirname(resolved_path)
            parent_display = "/" + os.path.relpath(parent_path, DICOM_DATA_ROOT) if parent_path != DICOM_DATA_ROOT else "/"
            if parent_display == "/.":
                parent_display = "/"

            items.append({
                "name": "..",
                "path": parent_display,
                "type": "directory",
                "is_parent": True
            })

        # List directory contents
        try:
            for item in sorted(os.listdir(resolved_path)):
                if item.startswith('.'):  # Skip hidden files
                    continue

                item_path = os.path.join(resolved_path, item)
                item_display_path = display_path.rstrip('/') + '/' + item
                if item_display_path.startswith('//'):
                    item_display_path = item_display_path[1:]

                if os.path.isdir(item_path):
                    # Count DICOM files in subdirectory for preview
                    dicom_count = 0
                    try:
                        for subitem in os.listdir(item_path):
                            subitem_path = os.path.join(item_path, subitem)
                            if os.path.isfile(subitem_path) and (
                                subitem.lower().endswith(('.dcm', '.dicom', '.ima')) or '.' not in subitem
                            ):
                                dicom_count += 1
                    except (PermissionError, OSError):
                        pass

                    items.append({
                        "name": item,
                        "path": item_display_path,
                        "type": "directory",
                        "is_parent": False,
                        "dicom_files": dicom_count
                    })
                elif item.lower().endswith(('.dcm', '.dicom', '.ima')) or ('.' not in item and os.path.isfile(item_path)):
                    # Verify it's actually a DICOM file
                    try:
                        pydicom.dcmread(item_path, stop_before_pixels=True)
                        items.append({
                            "name": item,
                            "path": item_display_path,
                            "type": "dicom_file",
                            "is_parent": False
                        })
                    except:
                        # Not a valid DICOM file, skip
                        continue

        except PermissionError:
            raise HTTPException(status_code=403, detail=f"Permission denied to access directory: {display_path}")

        return {
            "current_path": display_path,
            "resolved_path": resolved_path,
            "dicom_data_root": DICOM_DATA_ROOT,
            "items": items
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error browsing directory {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error browsing directory: {str(e)}")

@app.post("/api/load-series", response_model=LoadSeriesResponse)
async def load_series(series_uid: str = Query(..., description="Series UID to load")):
    if series_uid not in backend_cache.current_series_info: raise HTTPException(404, "Series UID not found.")
    backend_cache.clear_all_context()
    slices, volume = load_dicom_volume(backend_cache.current_series_info[series_uid]["files"])
    if slices is None: raise HTTPException(400, "Failed to load DICOM volume.")
    backend_cache.current_volume, backend_cache.current_slices = volume, slices
    meta = {"PatientID": getattr(slices[0], 'PatientID', 'N/A'), "PatientAge": getattr(slices[0], 'PatientAge', 'N/A')}
    return LoadSeriesResponse(status="success", shape=list(volume.shape), value_range=[float(volume.min()), float(volume.max())], metadata=meta)

@app.post("/api/generate-global-context")
async def generate_global_context_with_disk_cache(slice_index: int = Query(...), force_regenerate: bool = Query(False)):
    if backend_cache.current_volume is None: 
        raise HTTPException(status_code=400, detail="No DICOM series loaded.")
    
    series_uid = list(backend_cache.current_series_info.keys())[0]
    full_slice_np = backend_cache.current_volume[slice_index]
    
    # Check disk cache first
    if not force_regenerate:
        cached_context = disk_global_cache.get_cached_context(series_uid, slice_index, full_slice_np)
        if cached_context:
            return {"status": "cached", **cached_context}
    
    backend_cache.set_progress(f"Generating global anatomical map for slice {slice_index}...")
    try:
        slice_url = check_and_upload_slice_to_gcs(full_slice_np, f"{slice_index}_global", series_uid)
        if not slice_url: 
            raise HTTPException(500, "Failed to upload slice for global analysis.")
        
        prompt = """You are an expert radiologist AI with comprehensive anatomical knowledge. Your task is to perform an exhaustive analysis of the provided medical image and generate a detailed anatomical map in JSON format.

        **CRITICAL REQUIREMENTS:**

        1. **Coordinate System**: ALL bounding box coordinates must be normalized between 0.0 and 1.0 (top-left origin). Any coordinate outside this range is INVALID.

        2. **SLICE-SPECIFIC IDENTIFICATION** - Only identify structures that are ACTUALLY VISIBLE in this specific slice:

           **CRITICAL RULES:**
           - ONLY label structures you can clearly see in THIS slice
           - DO NOT include structures from adjacent slices or different anatomical levels
           - DO NOT use placeholder or template bounding boxes
           - Each bounding box must precisely outline the visible structure boundaries

           **STRUCTURE CATEGORIES TO CONSIDER (only if visible):**
           - **Bones**: Only the specific bones visible in this slice level
           - **Muscles**: Only muscle groups that cross this anatomical plane
           - **Organs**: Only organs/organ portions visible at this slice level
           - **Vessels**: Only vascular structures visible in this cross-section
           - **Fat/Soft Tissue**: Only the fat compartments and soft tissues present
           - **Other**: Any other structures clearly delineated in this slice

        3. **PATHOLOGY DETECTION**: Actively identify any abnormalities or unusual findings:
           - Any lesions, masses, or space-occupying structures
           - Structural abnormalities, trauma, or degenerative changes
           - Unusual densities, calcifications, or foreign materials
           - Anatomical variants, congenital anomalies, or developmental changes
           - Any finding that deviates from normal anatomy

        4. **PRECISION REQUIREMENTS**:
           - NO DUPLICATES: Each structure should appear only once
           - NO PLACEHOLDER COORDINATES: Every bounding box must be carefully drawn around actual visible margins
           - TIGHT BOUNDING BOXES: Coordinates must precisely follow the anatomical boundaries visible in the image
           - SLICE-APPROPRIATE STRUCTURES: Do not include muscles/organs from different anatomical levels

        5. **ENHANCED CATEGORIZATION** - Use these specific types:
           `bone`, `joint`, `muscle`, `organ`, `vessel`, `nerve`, `fat`, `fascia`, `ligament`,
           `lymph_node`, `lesion`, `fluid`, `calcification`, `foreign_body`, `abnormality`, `other`

        6. **CLINICAL DESCRIPTIONS**: Provide detailed, clinically relevant descriptions including:
           - Morphology (size, shape, density)
           - Position relative to other structures
           - Pathological findings with differential considerations
           - Recommendations for follow-up if indicated

        7. **COORDINATE VALIDATION**: Ensure ALL coordinates are between 0.0-1.0. Double-check before output.

        8. **JSON OUTPUT**: Single clean JSON object with "structures" array. No markdown, explanations, or text outside JSON.

        **EXAMPLE OUTPUT** (only structures visible in this specific slice):
        { 
        [ THESE EXAMPLES AND THE PLACEHOLDERS COORDINATES ARE STRICTLY GIVEN FOR SHOWING THE FORMAT. DO NOT USE THEM FOR PLACEHOLDER OUTPUT ]
        "structures": [
            {
            "label": "right femoral head",
            "bounding_box": {"x_min": 0.15, "y_min": 0.62, "x_max": 0.25, "y_max": 0.75},
            "type": "bone",
            "description": "Round femoral head with normal cortical margins and trabecular pattern."
            },
            {
            "label": "left acetabulum",
            "bounding_box": {"x_min": 0.72, "y_min": 0.58, "x_max": 0.82, "y_max": 0.68},
            "type": "bone",
            "description": "Hip socket with normal acetabular roof and rim. No joint space narrowing."
            },
            {
            "label": "right gluteus maximus",
            "bounding_box": {"x_min": 0.05, "y_min": 0.35, "x_max": 0.20, "y_max": 0.60},
            "type": "muscle",
            "description": "Large posterior hip muscle with normal bulk and homogeneous attenuation."
            }
        ]
        }

        ANALYZE THE IMAGE AND PROVIDE COMPREHENSIVE JSON OUTPUT WITH ALL VISIBLE ANATOMICAL STRUCTURES.
        """
        result = await call_medgamma_inference(slice_url, prompt)
        if "error" in result: 
            raise HTTPException(502, f"MedGamma API error: {result.get('details', result['error'])}")
        
        content = ""
        try:
            # Fixed parsing logic
            if not result.get('predictions'):
                logger.error(f"Unexpected API response received from MedGamma. Full response: {json.dumps(result, indent=2)}")
                raise HTTPException(500, "Invalid response from the analysis model: 'predictions' key is missing. Check server logs for the full API response.")
            
            prediction_content = result['predictions']
            
            if not prediction_content.get('choices') or not isinstance(prediction_content['choices'], list) or len(prediction_content['choices']) == 0:
                logger.error(f"API response contained predictions but no choices. Full response: {json.dumps(result, indent=2)}")
                raise HTTPException(500, "Model returned an empty response (no choices). This may be due to safety filters. Check server logs for the full API response.")
            
            content = prediction_content['choices'][0]['message']['content']

            json_str = content[content.find('{'):content.rfind('}')+1]
            anatomical_map = json.loads(json_str)

            if "structures" not in anatomical_map:
                raise ValueError("Missing 'structures' key in the parsed JSON response")

            # Validate and fix coordinate bounds for all structures
            valid_structures = []
            for structure in anatomical_map.get("structures", []):
                if "bounding_box" in structure and isinstance(structure["bounding_box"], dict):
                    bbox = structure["bounding_box"]
                    # Clamp coordinates to the valid range [0.0, 1.0]
                    bbox["x_min"] = max(0.0, min(1.0, float(bbox.get("x_min", 0.0))))
                    bbox["y_min"] = max(0.0, min(1.0, float(bbox.get("y_min", 0.0))))
                    bbox["x_max"] = max(0.0, min(1.0, float(bbox.get("x_max", 1.0))))
                    bbox["y_max"] = max(0.0, min(1.0, float(bbox.get("y_max", 1.0))))

                    # Ensure min values are less than max values
                    if bbox["x_min"] >= bbox["x_max"]:
                        bbox["x_max"] = min(1.0, bbox["x_min"] + 0.01) # Add a small offset
                    if bbox["y_min"] >= bbox["y_max"]:
                        bbox["y_max"] = min(1.0, bbox["y_min"] + 0.01)

                    structure["bounding_box"] = bbox
                    valid_structures.append(structure)
                else:
                    logger.warning(f"Skipping structure due to missing or invalid bounding box: {structure.get('label', 'N/A')}")

            anatomical_map["structures"] = valid_structures
            logger.info(f"Validated {len(valid_structures)} anatomical structures with proper coordinate bounds.")
            
            # Store in disk cache
            disk_global_cache.store_context(series_uid, slice_index, full_slice_np, anatomical_map)
            
            return {"status": "generated", **anatomical_map}
            
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to process model response: {e}\nResponse content was: {content}")
            raise HTTPException(500, f"Invalid structure in response from model: {e}")
    
    finally:
        backend_cache.set_progress("Idle")


class RectangleROIRequest(BaseModel):
    top_left_x: float = Field(..., ge=0, le=COORDINATE_SCALE, description="Top-left X coordinate")
    top_left_y: float = Field(..., ge=0, le=COORDINATE_SCALE, description="Top-left Y coordinate")
    bottom_right_x: float = Field(..., ge=0, le=COORDINATE_SCALE, description="Bottom-right X coordinate")
    bottom_right_y: float = Field(..., ge=0, le=COORDINATE_SCALE, description="Bottom-right Y coordinate")

class CircleROIRequest(BaseModel):
    center_x: float = Field(..., ge=0, le=COORDINATE_SCALE, description="Center X coordinate")
    center_y: float = Field(..., ge=0, le=COORDINATE_SCALE, description="Center Y coordinate")
    radius: float = Field(..., gt=0, le=COORDINATE_SCALE//2, description="Radius in pixels")

# =============================================================================
# RECTANGLE ROI ANALYSIS ENDPOINT
# =============================================================================
# =============================================================================
# SIMPLIFIED RECTANGLE ROI ANALYSIS ENDPOINT
# =============================================================================
@app.post("/api/analyze-roi-region/rectangle")
async def analyze_rectangle_roi_endpoint(
    slice_index: int = Query(..., description="Slice index to analyze"),
    top_left_x: float = Query(..., ge=0, le=COORDINATE_SCALE, description="Top-left X coordinate"),
    top_left_y: float = Query(..., ge=0, le=COORDINATE_SCALE, description="Top-left Y coordinate"),
    bottom_right_x: float = Query(..., ge=0, le=COORDINATE_SCALE, description="Bottom-right X coordinate"),
    bottom_right_y: float = Query(..., ge=0, le=COORDINATE_SCALE, description="Bottom-right Y coordinate")
):
    """Analyze rectangular ROI region."""
    
    if backend_cache.is_processing: 
        raise HTTPException(429, "Analysis in progress.")
    if backend_cache.current_volume is None: 
        raise HTTPException(400, "No DICOM series loaded.")
    if not all([local_medsam.is_ready, local_analyzer.is_ready]): 
        raise HTTPException(503, "Local AI models are not ready.")
    
    backend_cache.is_processing = True
    try:
        # Create rectangular ROI region
        roi_region = RectangularROI(
            Point(top_left_x, top_left_y),
            Point(bottom_right_x, bottom_right_y)
        )
        roi_description = f"Rectangle from ({top_left_x}, {top_left_y}) to ({bottom_right_x}, {bottom_right_y})"
        
        roi_request_dict = {
            "top_left_x": top_left_x,
            "top_left_y": top_left_y,
            "bottom_right_x": bottom_right_x,
            "bottom_right_y": bottom_right_y
        }
        
        # Call shared analysis function
        return await _analyze_roi_shared(slice_index, roi_region, roi_description, "rectangle", roi_request_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in rectangle ROI analysis: {str(e)}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        backend_cache.is_processing = False
        backend_cache.set_progress("Idle")

# =============================================================================
# SIMPLIFIED CIRCLE ROI ANALYSIS ENDPOINT
# =============================================================================
@app.post("/api/analyze-roi-region/circle")
async def analyze_circle_roi_endpoint(
    slice_index: int = Query(..., description="Slice index to analyze"),
    center_x: float = Query(..., ge=0, le=COORDINATE_SCALE, description="Center X coordinate"),
    center_y: float = Query(..., ge=0, le=COORDINATE_SCALE, description="Center Y coordinate"),
    radius: float = Query(..., gt=0, le=COORDINATE_SCALE//2, description="Radius in pixels")
):
    """Analyze circular ROI region."""
    
    if backend_cache.is_processing: 
        raise HTTPException(429, "Analysis in progress.")
    if backend_cache.current_volume is None: 
        raise HTTPException(400, "No DICOM series loaded.")
    if not all([local_medsam.is_ready, local_analyzer.is_ready]): 
        raise HTTPException(503, "Local AI models are not ready.")
    
    backend_cache.is_processing = True
    try:
        # Create circular ROI region
        roi_region = CircularROI(
            Point(center_x, center_y),
            radius
        )
        roi_description = f"Circle at ({center_x}, {center_y}) with radius {radius}"
        
        roi_request_dict = {
            "center_x": center_x,
            "center_y": center_y,
            "radius": radius
        }
        
        # Call shared analysis function
        return await _analyze_roi_shared(slice_index, roi_region, roi_description, "circle", roi_request_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in circle ROI analysis: {str(e)}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        backend_cache.is_processing = False
        backend_cache.set_progress("Idle")
# =============================================================================
# SHARED ROI ANALYSIS LOGIC
# =============================================================================
# =============================================================================
# CORRECTED SHARED ROI ANALYSIS LOGIC - NO MEDGAMMA CALLS
# =============================================================================
async def _analyze_roi_shared(slice_index: int, roi_region, roi_description: str, roi_type: str, roi_request_dict: dict):
    """Shared logic for both rectangle and circle ROI analysis - uses only global context and local analysis."""
    
    backend_cache.set_progress("Step 1: Loading slice and anatomical context...")
    
    # Get global context from disk cache
    series_uid = list(backend_cache.current_series_info.keys())[0]
    full_slice_np = backend_cache.current_volume[slice_index]
    anatomical_map = disk_global_cache.get_cached_context(series_uid, slice_index, full_slice_np)
    
    if not anatomical_map:
        raise HTTPException(400, f"No global anatomical context found for slice {slice_index}. Please generate global context first using /api/generate-global-context?slice_index={slice_index}")
    
    backend_cache.set_progress("Step 2: Analyzing anatomical overlap...")
    
    # Analyze what anatomical structures the ROI overlaps with using global context
    overlap_analysis = analyze_roi_anatomical_overlap(roi_region, anatomical_map)
    
    # Build detailed anatomical context
    if overlap_analysis['overlapping_structures']:
        primary_structure = overlap_analysis['overlapping_structures'][0]
        anatomical_context = f"ROI is primarily located in: {primary_structure['structure']['label']} ({primary_structure['roi_coverage']:.1f}% of ROI)"
        
        # Add secondary structures if present
        if len(overlap_analysis['overlapping_structures']) > 1:
            secondary_structures = overlap_analysis['overlapping_structures'][1:3]  # Top 2 additional
            secondary_names = [s['structure']['label'] for s in secondary_structures]
            anatomical_context += f". Also overlaps: {', '.join(secondary_names)}"
        
        # Add anatomical details from the structures
        anatomical_details = []
        for struct_info in overlap_analysis['overlapping_structures'][:3]:  # Top 3 structures
            structure = struct_info['structure']
            details = f"- {structure['label']}: {structure.get('description', 'Normal anatomy')}"
            if 'type' in structure:
                details += f" (Type: {structure['type']})"
            anatomical_details.append(details)
        
        full_anatomical_context = f"{anatomical_context}\n\nAnatomical Details:\n" + "\n".join(anatomical_details)
    else:
        anatomical_context = f"ROI is located in an uncharacterized region of the image"
        full_anatomical_context = anatomical_context
    
    backend_cache.set_progress("Step 3: Segmenting ROI...")
    
    # Use ROI center for segmentation
    roi_center = roi_region.get_center()
    h, w = full_slice_np.shape
    vol_x, vol_y = int(roi_center.x * (w / COORDINATE_SCALE)), int(roi_center.y * (h / COORDINATE_SCALE))
    
    # Validate coordinates are within bounds
    vol_x = max(0, min(vol_x, w - 1))
    vol_y = max(0, min(vol_y, h - 1))
    
    mask = local_medsam.get_mask(full_slice_np, (vol_x, vol_y))
    if mask is None or np.sum(mask) == 0: 
        raise HTTPException(400, f"Segmentation failed at coordinates ({vol_x}, {vol_y}). Try a different ROI location.")
    
    backend_cache.set_progress("Step 4: Analyzing ROI properties...")
    
    # HU statistics for the segmented region
    segmented_pixels = full_slice_np[mask == 1]
    hu_stats = {
        "mean_hu": float(np.mean(segmented_pixels)),
        "std_hu": float(np.std(segmented_pixels)),
        "min_hu": float(np.min(segmented_pixels)),
        "max_hu": float(np.max(segmented_pixels)),
        "pixel_count": int(np.sum(mask))
    }
    
    # Crop ROI for visual analysis (but DON'T send to MedGamma)
    props = regionprops(mask)[0]
    y_min, x_min, y_max, x_max = props.bbox
    roi_cropped = full_slice_np[y_min:y_max, x_min:x_max]
    
    # Ensure ROI is not too small
    if roi_cropped.size < 100:  # Less than 10x10 pixels
        raise HTTPException(400, "Segmented ROI is too small for reliable analysis. Try selecting a larger region.")
    
    # Local AI analysis only
    roi_image_pil = Image.fromarray((roi_cropped * 255 / roi_cropped.max()).astype(np.uint8)).convert("RGB")
    local_result = local_analyzer.analyze_roi(roi_image_pil)
    if "error" in local_result: 
        raise HTTPException(500, f"Local AI analysis failed: {local_result['error']}")
    
    backend_cache.set_progress("Step 5: Generating diagnosis based on anatomical context...")
    
    # Generate comprehensive diagnosis based ONLY on local analysis and anatomical context
    # NO MedGamma calls - use the anatomical context and local analysis to create diagnosis
    
    # Build comprehensive diagnosis based on anatomical context
    diagnosis_text = f"""Based on the anatomical context and local analysis:

ANATOMICAL LOCATION: {overlap_analysis['primary_anatomy']}
REGION ANALYSIS: {full_anatomical_context}

QUANTITATIVE FINDINGS:
- HU Values: Mean={hu_stats['mean_hu']:.1f}, Range=[{hu_stats['min_hu']:.0f}, {hu_stats['max_hu']:.0f}]
- This suggests tissue density characteristics consistent with the anatomical location
- Segmented area: {hu_stats['pixel_count']} pixels


CONTEXTUAL INTERPRETATION:
Given the anatomical location in {overlap_analysis['primary_anatomy']}, the HU values and visual characteristics are """

    # Add context-specific interpretation based on primary anatomy
    if overlap_analysis['overlapping_structures']:
        primary_struct = overlap_analysis['overlapping_structures'][0]['structure']
        struct_type = primary_struct.get('type', 'tissue')
        
        if 'bone' in struct_type.lower():
            diagnosis_text += "consistent with osseous tissue. The HU values should be elevated (>300 HU) for normal bone density."
        elif 'soft tissue' in struct_type.lower() or 'muscle' in primary_struct['label'].lower():
            diagnosis_text += "consistent with soft tissue density. Normal soft tissue typically ranges from 10-40 HU."
        elif 'lung' in primary_struct['label'].lower():
            diagnosis_text += "consistent with pulmonary tissue. Normal aerated lung should show very low HU values (-500 to -900 HU)."
        elif 'vessel' in primary_struct['label'].lower():
            diagnosis_text += "consistent with vascular structure. May show contrast enhancement if contrast was administered."
        else:
            diagnosis_text += f"consistent with {struct_type} in this anatomical region."
    else:
        diagnosis_text += "within normal expected range for this region."
    
    # Add recommendations based on findings
    diagnosis_text += f"""

ASSESSMENT:
- Primary finding: {local_result['analysis']['label']} in {overlap_analysis['primary_anatomy']}
- Anatomical correlation: Appropriate for the selected region


RECOMMENDATIONS:
- Findings are consistent with the anatomical location
- Consider clinical correlation if symptoms are present
- Compare with prior imaging if available"""
    
    # Create structured final result similar to MedGamma format but generated locally
    final_result = {
        "analysis": diagnosis_text,
        "primary_finding": local_result['analysis']['label'],
        "anatomical_location": overlap_analysis['primary_anatomy'],
        "hu_interpretation": "Within expected range" if -100 <= hu_stats['mean_hu'] <= 100 else "Atypical density values",
        "source": "Local analysis with anatomical context"
    }
    
    backend_cache.set_progress("Completed - No cloud analysis required")
    
    return {
        "status": "success",
        "slice_index": slice_index,
        "roi_type": roi_type,
        "roi_description": roi_description,
        "roi_region": roi_request_dict,
        "anatomical_analysis": {
            "primary_anatomy": overlap_analysis['primary_anatomy'],
            "overlapping_structures": overlap_analysis['overlapping_structures'],
            "coverage_analysis": overlap_analysis['coverage_analysis'],
            "spatial_context": anatomical_context
        },
        "quantitative_analysis": {
            "hu_statistics": hu_stats,
            "visual_classification": local_result['analysis'],
            "segmentation_success": True,
            "roi_area_pixels": hu_stats['pixel_count']
        },
        "final_diagnosis": final_result,
        "processing_metadata": {
            "segmentation_center": {"x": vol_x, "y": vol_y},
            "roi_bounds": {"y_min": y_min, "x_min": x_min, "y_max": y_max, "x_max": x_max},
            "anatomical_structures_found": len(overlap_analysis['overlapping_structures']),
            "analysis_method": "Local analysis with cached global context - no cloud calls"
        }
    }

@app.get("/api/analyze-whole-image-preview/{slice_index}")
async def preview_whole_image_analysis(
    slice_index: int = Path(..., description="Slice index"),
    analysis_type: str = Query("comprehensive", description="Analysis type preview"),
    show_regions: bool = Query(True, description="Show detected regions as overlay")
):
    """
    Get a preview of whole image analysis regions as PNG
    
    This endpoint shows what regions would be detected without running full analysis
    Useful for parameter tuning before running the expensive full analysis
    
    Parameters:
    - slice_index: Which slice to preview
    - analysis_type: Type of analysis to preview
    - show_regions: Whether to overlay detected regions
    
    Returns:
    - PNG image showing detected regions
    """
    try:
        if backend_cache.current_volume is None:
            raise HTTPException(status_code=400, detail="No volume loaded")
        
        volume = backend_cache.current_volume
        
        if slice_index < 0 or slice_index >= volume.shape[0]:
            raise HTTPException(status_code=400, detail="Invalid slice index")
        
        slice_2d = volume[slice_index, :, :]
        
        # Create preview image with better normalization
        if np.ptp(slice_2d) > 0:
            # Use percentile-based normalization for better contrast
            p2, p98 = np.percentile(slice_2d, (2, 98))
            normalized = np.clip(slice_2d, p2, p98)
            normalized = (normalized - p2) / (p98 - p2) * 255
        else:
            normalized = np.zeros_like(slice_2d)

        preview_image = normalized.astype(np.uint8)

        # Resize to standard IMAGE_TARGET_SIZE if not already
        if preview_image.shape != IMAGE_TARGET_SIZE:
            preview_image = cv2.resize(preview_image, IMAGE_TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        
        if show_regions:
            # Quick region detection for preview
            if analysis_type == "grid":
                # Show grid overlay
                h, w = slice_2d.shape
                grid_size = 32
                rows_per_cell = h // grid_size
                cols_per_cell = w // grid_size
                
                # Draw grid lines
                for i in range(0, h, rows_per_cell):
                    preview_image[i:i+1, :] = 128  # Horizontal lines
                for j in range(0, w, cols_per_cell):
                    preview_image[:, j:j+1] = 128  # Vertical lines
                    
            elif analysis_type == "intensity_regions":
                # Show intensity-based regions
                try:
                    intensity_thresholds = np.percentile(slice_2d, [25, 50, 75])
                    for i, threshold in enumerate(intensity_thresholds):
                        mask = slice_2d > threshold
                        preview_image[mask] = min(255, preview_image[mask] + 30)
                except:
                    pass
                    
            else:  # comprehensive
                # Show Otsu thresholding result
                try:
                    from skimage import filters
                    threshold = filters.threshold_otsu(slice_2d)
                    mask = slice_2d > threshold
                    preview_image[mask] = min(255, preview_image[mask] + 50)
                except:
                    pass
        
        # Convert to PIL and return as high-quality PNG
        pil_image = Image.fromarray(preview_image, mode='L')
        img_buffer = BytesIO()
        pil_image.save(
            img_buffer,
            format='PNG',
            optimize=True,
            compress_level=6
        )
        img_buffer.seek(0)

        return StreamingResponse(
            img_buffer,
            media_type="image/png",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in preview: {e}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")








# Legacy endpoint for backward compatibility
@app.post("/api/analyze-roi")
async def analyze_roi_endpoint(slice_index: int = Query(...), canvas_x: float = Query(...), canvas_y: float = Query(...)):
    if backend_cache.is_processing: raise HTTPException(429, "Analysis in progress.")
    if backend_cache.current_volume is None: raise HTTPException(400, "No DICOM series loaded.")
    if not all([local_medsam.is_ready, local_analyzer.is_ready]): raise HTTPException(503, "Local AI models are not ready.")
    
    backend_cache.is_processing = True
    try:
        backend_cache.set_progress("Step 1: Determining anatomical location...")
        
        # Get global context from disk cache
        series_uid = list(backend_cache.current_series_info.keys())[0]
        full_slice_np = backend_cache.current_volume[slice_index]
        anatomical_map = disk_global_cache.get_cached_context(series_uid, slice_index, full_slice_np)
        
        if not anatomical_map:
            anatomical_map = {}
        
        overlapping_structures = find_overlapping_structures(canvas_x, canvas_y, anatomical_map)
        anatomical_context = f"Clicked within '{overlapping_structures[0]['structure']['label']}'." if overlapping_structures else "Clicked in an uncharacterized region."
        
        backend_cache.set_progress("Step 2: Segmenting ROI...")
        h, w = full_slice_np.shape
        vol_x, vol_y = int(canvas_x * (w / COORDINATE_SCALE)), int(canvas_y * (h / COORDINATE_SCALE))
        mask = local_medsam.get_mask(full_slice_np, (vol_x, vol_y))
        if mask is None or np.sum(mask) == 0: raise HTTPException(400, "Segmentation failed.")
        
        backend_cache.set_progress("Step 3: Analyzing ROI properties...")
        hu_stats = {"mean_hu": float(np.mean(full_slice_np[mask==1]))}
        props = regionprops(mask)[0]
        roi_cropped = full_slice_np[props.bbox[0]:props.bbox[2], props.bbox[1]:props.bbox[3]]
        local_result = local_analyzer.analyze_roi(Image.fromarray(roi_cropped).convert("RGB"))
        if "error" in local_result: raise HTTPException(500, local_result['error'])
        
        backend_cache.set_progress("Step 4: Searching knowledge base...")
        retrieved_cases = vector_db_manager.search(local_result["embedding"], k=3)
        
        backend_cache.set_progress("Step 5: Generating final diagnosis...")
        roi_url = check_and_upload_slice_to_gcs(roi_cropped, f"{slice_index}_roi_{vol_x}_{vol_y}", series_uid)
        if not roi_url: raise HTTPException(500, "Failed to upload ROI")
        
        prompt = f"Analyze this ROI. Spatial Context: {anatomical_context}. HU mean: {hu_stats['mean_hu']:.0f}. Visual analysis: '{local_result['analysis']['label']}'. Provide a primary diagnosis and assessment."
        final_result = await call_medgamma_inference(roi_url, prompt)
        
        return {"status": "success", "final_analysis": final_result, "context": {"spatial": anatomical_context, "hu": hu_stats, "visual": local_result['analysis'], "retrieved_cases": retrieved_cases}}
    finally:
        backend_cache.is_processing = False
        backend_cache.set_progress("Idle")


@app.get("/api/get-global-context")
async def get_global_context(slice_index: int = Query(...)):
    series_uid = list(backend_cache.current_series_info.keys())[0]
    full_slice_np = backend_cache.current_volume[slice_index]
    context = disk_global_cache.get_cached_context(series_uid, slice_index, full_slice_np)
    if not context: raise HTTPException(404, f"Global context for slice {slice_index} not found.")
    return context



@app.delete("/api/cache/global-context/clear")
async def clear_global_cache():
    """Clear global context cache."""
    try:
        disk_global_cache.clear_cache()
        return {"status": "Global context cache cleared"}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/api/cache/global-context/clear/{series_uid}")
async def clear_series_cache(series_uid: str):
    """Clear cache for a specific series."""
    try:
        disk_global_cache.clear_cache(series_uid)
        return {"status": f"Cache cleared for series {series_uid}"}
    except Exception as e:
        return {"error": str(e)}



@app.post("/api/system/clear-all-cache")
async def clear_all_system_cache():
    """Clear all system caches"""
    try:
        # Clear global context cache
        disk_global_cache.clear_cache()
        
        # Clear backend cache
        backend_cache.clear_all_context()
        
        # Force garbage collection
        gc.collect()
        
        return {"status": "All caches cleared successfully"}
    except Exception as e:
        return {"error": str(e)}



@app.post("/api/system/reload-models")
async def reload_ai_models():
    """Reload AI models"""
    try:
        # Reload MedSAM
        local_medsam.is_ready = False
        local_medsam.load_model()

        # Reload ROI Analyzer
        local_analyzer.is_ready = False
        local_analyzer.load_model()

        # Reload vector database
        vector_db_manager.load_database()

        return {
            "status": "Models reloaded",
            "medsam_ready": local_medsam.is_ready,
            "analyzer_ready": local_analyzer.is_ready,
            "vector_db_entries": vector_db_manager.index.ntotal if vector_db_manager.index else 0
        }
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# 3D MESH GENERATION ENDPOINTS
# =============================================================================

@app.post("/api/3d/generate-mesh/{series_uid}")
async def generate_3d_mesh(
    background_tasks: BackgroundTasks,
    series_uid: str = Path(..., description="Series UID to process"),
    threshold_hu: int = Query(250, ge=-1000, le=3000, description="Threshold in Hounsfield Units for isosurface extraction"),
    decimation_factor: float = Query(0.9, ge=0.1, le=0.99, description="Mesh reduction factor (0.1-0.99)")
):
    """
    Generate 3D mesh from DICOM series using marching cubes algorithm.

    This endpoint starts a background task to process the DICOM series into a 3D mesh.
    The mesh is optimized for web viewing and exported as GLB format.

    Parameters:
    - series_uid: The series to process (must be loaded in current session)
    - threshold_hu: Hounsfield Unit threshold for isosurface (250 good for bone)
    - decimation_factor: How much to reduce mesh complexity (0.9 = 90% reduction)

    Returns:
    - job_id: Use this to check status and get results
    - status_endpoint: URL to poll for progress updates
    """
    start_time = time.time()

    # Track request metrics
    with metrics_lock:
        disease_metrics["total_requests"] += 1
        disease_metrics["last_request_time"] = datetime.now()
        disease_metrics["active_sessions"] += 1

    try:
        # Validate series exists in current session
        if not backend_cache.current_series_info or series_uid not in backend_cache.current_series_info:
            raise HTTPException(404, f"Series '{series_uid}' not found in current session. Please scan and load a series first.")

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Initialize job status
        mesh_status_manager.set_status(job_id, "queued", 0, "Job queued for processing...")

        # Start background processing
        background_tasks.add_task(
            process_dicom_to_3d_mesh,
            series_uid,
            threshold_hu,
            decimation_factor,
            job_id,
            mesh_status_manager
        )

        # Track success metrics
        response_time = (time.time() - start_time) * 1000
        with metrics_lock:
            disease_metrics["successful_requests"] += 1
            disease_metrics["response_times"].append(response_time)
            disease_metrics["active_sessions"] = max(0, disease_metrics["active_sessions"] - 1)
            # Keep only last 100 response times
            if len(disease_metrics["response_times"]) > 100:
                disease_metrics["response_times"] = disease_metrics["response_times"][-100:]

        logger.info(f"Started 3D mesh generation for series {series_uid} with job ID {job_id}")

        return {
            "job_id": job_id,
            "status": "queued",
            "series_uid": series_uid,
            "parameters": {
                "threshold_hu": threshold_hu,
                "decimation_factor": decimation_factor
            },
            "status_endpoint": f"/api/3d/status/{job_id}",
            "estimated_duration": "1-3 minutes depending on series size"
        }

    except HTTPException:
        with metrics_lock:
            disease_metrics["failed_requests"] += 1
            disease_metrics["active_sessions"] = max(0, disease_metrics["active_sessions"] - 1)
        raise
    except Exception as e:
        with metrics_lock:
            disease_metrics["failed_requests"] += 1
            disease_metrics["active_sessions"] = max(0, disease_metrics["active_sessions"] - 1)
        logger.error(f"Error starting 3D mesh generation: {e}")
        raise HTTPException(500, f"Failed to start 3D mesh generation: {str(e)}")

@app.get("/api/3d/status/{job_id}")
async def get_3d_mesh_status(job_id: str = Path(..., description="Job ID from mesh generation request")):
    """
    Get the status of a 3D mesh generation job.

    Returns:
    - status: queued, processing, completed, or failed
    - progress: 0.0 to 1.0 indicating completion percentage
    - message: Human-readable status message
    - mesh_url: Download URL (only when completed)
    - error: Error message (only when failed)
    """
    status = mesh_status_manager.get_status(job_id)

    if not status:
        raise HTTPException(404, f"Job ID '{job_id}' not found. Jobs expire after completion.")

    return status

@app.get("/api/meshes/{mesh_file}")
async def download_mesh_file(mesh_file: str = Path(..., description="Mesh filename (e.g., job-id.glb)")):
    """
    Download a generated 3D mesh file.

    The mesh is in GLB format, optimized for web viewing with Three.js or similar libraries.
    """
    # Security: Only allow .glb files and prevent directory traversal
    if not mesh_file.endswith('.glb') or '/' in mesh_file or '\\' in mesh_file:
        raise HTTPException(400, "Invalid mesh filename")

    mesh_path = os.path.join("static/meshes", mesh_file)

    if not os.path.exists(mesh_path):
        raise HTTPException(404, f"Mesh file '{mesh_file}' not found")

    # Track cache operation
    with metrics_lock:
        disease_metrics["cache_operations"] += 1

    return FileResponse(
        mesh_path,
        media_type="model/gltf-binary",
        filename=mesh_file,
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
            "Content-Disposition": f"attachment; filename={mesh_file}"
        }
    )

@app.get("/api/3d/list-available-series")
async def list_available_series_for_3d():
    """
    List all DICOM series available in the current session for 3D mesh generation.

    Returns series information including slice count and basic metadata.
    """
    if not backend_cache.current_series_info:
        return {"available_series": [], "message": "No DICOM series loaded. Please scan a directory first."}

    series_list = []
    for uid, info in backend_cache.current_series_info.items():
        series_list.append({
            "series_uid": uid,
            "description": info.get("description", "N/A"),
            "slice_count": info.get("slices", 0),
            "patient_id": info.get("patient_id", "N/A"),
            "modality": info.get("modality", "N/A"),
            "suitable_for_3d": info.get("slices", 0) >= 10  # Need minimum slices for 3D
        })

    return {
        "available_series": series_list,
        "total_series": len(series_list),
        "recommendation": "CT series with 50+ slices work best for 3D mesh generation"
    }

@app.delete("/api/3d/cleanup-meshes")
async def cleanup_old_mesh_files(
    max_age_hours: int = Query(24, ge=1, le=168, description="Remove mesh files older than this many hours")
):
    """
    Clean up old mesh files to free disk space.

    Parameters:
    - max_age_hours: Remove files older than this (default: 24 hours)
    """
    try:
        mesh_dir = "static/meshes"
        if not os.path.exists(mesh_dir):
            return {"message": "No mesh directory found", "files_removed": 0}

        cutoff_time = time.time() - (max_age_hours * 3600)
        files_removed = 0

        for filename in os.listdir(mesh_dir):
            if filename.endswith('.glb'):
                file_path = os.path.join(mesh_dir, filename)
                if os.path.getmtime(file_path) < cutoff_time:
                    os.remove(file_path)
                    files_removed += 1

        return {
            "message": f"Cleanup completed",
            "files_removed": files_removed,
            "max_age_hours": max_age_hours
        }

    except Exception as e:
        logger.error(f"Error during mesh cleanup: {e}")
        raise HTTPException(500, f"Cleanup failed: {str(e)}")

# =============================================================================
# DATA EXPORT ENDPOINTS
# =============================================================================
@app.get("/api/export/global-context/{slice_index}")
async def export_global_context(slice_index: int):
    """Export global context for a specific slice"""
    try:
        if backend_cache.current_volume is None:
            raise HTTPException(400, "No DICOM series loaded")
        
        series_uid = list(backend_cache.current_series_info.keys())[0]
        full_slice_np = backend_cache.current_volume[slice_index]
        context = disk_global_cache.get_cached_context(series_uid, slice_index, full_slice_np)
        
        if not context:
            raise HTTPException(404, "Global context not found for this slice")
        
        # Add metadata
        export_data = {
            "slice_index": slice_index,
            "series_uid": series_uid,
            "export_timestamp": datetime.now().isoformat(),
            "global_context": context
        }
        
        return export_data
    except Exception as e:
        raise HTTPException(500, str(e))



# =============================================================================
# WEBHOOKS AND NOTIFICATIONS
# =============================================================================
@app.post("/api/webhooks/analysis-complete")
async def analysis_complete_webhook(
    webhook_url: str = Body(..., description="URL to send completion notification"),
    analysis_id: str = Body(..., description="Analysis ID"),
    results: Dict[str, Any] = Body(..., description="Analysis results")
):
    """Send webhook notification when analysis is complete"""
    try:
        payload = {
            "analysis_id": analysis_id,
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        response = requests.post(webhook_url, json=payload, timeout=30)
        response.raise_for_status()
        
        return {"status": "Webhook sent successfully"}
    except requests.exceptions.RequestException as e:
        raise HTTPException(500, f"Webhook failed: {str(e)}")

# =============================================================================
# SERVER ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Log startup information
    logger.info("🚀 Starting Standalone Medical AI Backend")
    logger.info(f"🔌 Server will be available at: http://0.0.0.0:6500")
    logger.info(f"📚 API Documentation: http://localhost:6500/docs")
    logger.info(f"🏥 Health Check: http://localhost:6500/api/health")
    logger.info("📋 Logging is configured for console output including Docker containers")

    # Run the server with enhanced logging
    uvicorn.run(
        "final:app",
        host="0.0.0.0",
        port=6500,
        reload=True,
        log_level="info",
        access_log=True,
        use_colors=True if sys.stdout.isatty() else False  # Auto-detect terminal for colors
    )

# =============================================================================
# IMPORTS
# =============================================================================
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
from collections import defaultdict
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union, Literal, Optional, Tuple, NamedTuple

# Third-party Libraries
import uvicorn
import pydicom
import numpy as np
import cv2
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, validator, Field
from PIL import Image

# Google Cloud Libraries
from google.cloud import storage
import google.auth
import google.auth.transport.requests

# Transformers for MedSigLIP
from transformers import AutoProcessor, AutoModel

# MedSAM model components
# Assuming these are custom modules available in the specified path
from Swin_LiteMedSAM.models.mask_decoder import MaskDecoder_Prompt
from Swin_LiteMedSAM.models.prompt_encoder import PromptEncoder
from Swin_LiteMedSAM.models.swin import SwinTransformer
from Swin_LiteMedSAM.models.transformer import TwoWayTransformer
from skimage.measure import regionprops



from pydantic import BaseModel, Field
from typing import Union, Literal
# RAG - Vector Database Library
import faiss

# =============================================================================
# CONFIGURATION AND APP INITIALIZATION
# =============================================================================


# Add this import at the top with your other imports
from fastapi.middleware.cors import CORSMiddleware

# Add this right after your app = FastAPI(...) line (around line 42)



load_dotenv()

# Enhanced logging configuration for console output (including Docker)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Ensures output to stdout/stderr
    ]
)

# Set uvicorn logging to also output to console
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.INFO)

# Get main logger
logger = logging.getLogger(__name__)

# Force logging to be unbuffered for Docker
import sys
sys.stdout.flush()
sys.stderr.flush()

# =============================================================================
# GLOBAL CONFIGURATION CONSTANTS
# =============================================================================
# Canvas/Coordinate system configuration
CANVAS_WIDTH = 512
CANVAS_HEIGHT = 512
CANVAS_SIZE = (CANVAS_WIDTH, CANVAS_HEIGHT)

# Image processing configuration
IMAGE_TARGET_SIZE = (512, 512)  # Standard size for all image operations
COORDINATE_SCALE = 512  # Scale for converting between canvas and volume coordinates

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






# =============================================================================
# ROI REGION HANDLING CLASSES
# =============================================================================
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

# =============================================================================
# ENHANCED ROI REQUEST MODELS
# =============================================================================
from pydantic import BaseModel, Field
from typing import Union, Literal



# =============================================================================
# DISK-BASED GLOBAL CONTEXT CACHING
# =============================================================================
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

# =============================================================================
# RAG - VECTOR DATABASE MODULE (FAISS)
# =============================================================================
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

# =============================================================================
# LOCAL AI ENGINES
# =============================================================================
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
    
    def load_model(self):
        try:
            logger.info(f"🔄 Starting MedSigLIP model loading onto {self.device}...")
            model_name = "google/medsiglip-448"
            
            logger.info(f"📥 Downloading/loading MedSigLIP model: {model_name}...")
            self.model = AutoModel.from_pretrained(model_name)
            logger.info("✅ MedSigLIP model loaded from HuggingFace")
            
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
            analysis = {"label": self.candidate_labels[np.argmax(probs)], "confidence": float(np.max(probs))}
            return {"analysis": analysis, "embedding": image_embeds}
        except Exception as e: return {"error": f"Local analysis failed: {e}"}

# =============================================================================
# CACHE AND UTILITIES
# =============================================================================
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
                    "content": [{"type": "text", "text": "You are a world-class radiologist."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": signed_url}}
                    ]
                }
            ]
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
                timeout=180
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
        logger.error(f"API request timed out after 180 seconds: {timeout_err}")
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
    s_map = defaultdict(list)
    for root, _, files in os.walk(os.path.expanduser(dir_path)):
        for f in files:
            try:
                ds = pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True)
                if 'SeriesInstanceUID' in ds: s_map[ds.SeriesInstanceUID].append(os.path.join(root, f))
            except Exception: continue
    s_info = {}
    for uid, files in s_map.items():
        try:
            ds = pydicom.dcmread(files[0])
            s_info[uid] = {"uid": uid, "description": getattr(ds, "SeriesDescription", "N/A"), "slices": len(files), "patient_id": getattr(ds, 'PatientID', 'N/A'), "modality": getattr(ds, 'Modality', 'N/A'), "files": files}
        except Exception: continue
    return s_info

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
# INITIALIZE COMPONENTS
# =============================================================================
local_medsam = LocalMedSAM()
local_analyzer = LocalROIAnalyzer()
vector_db_manager = VectorDBManager()
backend_cache = BackendCache()
disk_global_cache = DiskBasedGlobalContextCache()

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

@app.get("/api/scan-directory", response_model=List[SeriesInfo])
async def scan_directory(directory: str = Query(..., description="Directory path to scan")):
    series_info = scan_for_series(directory)
    backend_cache.current_series_info = series_info
    return [SeriesInfo(**info, display_name=f"{info['patient_id']}-{info['modality']}({info['slices']})") for _, info in series_info.items()]

@app.get("/api/browse-directories")
async def browse_directories(path: str = Query("/", description="Directory path to browse")):
    """Browse directories and files for DICOM directory selection"""
    import os
    try:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Path does not exist")

        if not os.path.isdir(path):
            raise HTTPException(status_code=400, detail="Path is not a directory")

        items = []

        # Add parent directory if not root
        if path != "/" and os.path.dirname(path) != path:
            items.append({
                "name": "..",
                "path": os.path.dirname(path),
                "type": "directory",
                "is_parent": True
            })

        # List directory contents
        try:
            for item in sorted(os.listdir(path)):
                if item.startswith('.'):  # Skip hidden files
                    continue

                item_path = os.path.join(path, item)

                if os.path.isdir(item_path):
                    items.append({
                        "name": item,
                        "path": item_path,
                        "type": "directory",
                        "is_parent": False
                    })
                elif item.lower().endswith(('.dcm', '.dicom')):
                    items.append({
                        "name": item,
                        "path": item_path,
                        "type": "dicom_file",
                        "is_parent": False
                    })
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied to access directory")

        return {
            "current_path": path,
            "items": items
        }

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
        
        prompt = """Analyze this CT image and create a precise JSON anatomical map. Use a top-left (0.0, 0.0) to bottom-right (1.0, 1.0) coordinate system. Provide tight bounding boxes for major bones, organs, and vessels.
        
        Example format:
        {
          "structures": [
            {"label": "right clavicle", "bounding_box": {"x_min": 0.123, "y_min": 0.456, "x_max": 0.234, "y_max": 0.567}, "type": "bone", "description": "Normal morphology"}
          ]
        }"""

        result = await call_medgamma_inference(slice_url, prompt)
        if "error" in result: 
            raise HTTPException(502, f"MedGamma API error: {result.get('details', result['error'])}")
        
        content = ""
        try:
            # Fixed parsing logic
            if not result.get('predictions'):
                logger.error(f"Unexpected API response received from MedGamma. Full response: {json.dumps(result, indent=2)}")
                raise HTTPException(500, "Invalid response from the analysis model: 'predictions' key is missing. Check server logs for the full API response.")
            
            prediction_content = result['predictions']  # predictions is a dict, not a list
            
            if not prediction_content.get('choices') or not isinstance(prediction_content['choices'], list) or len(prediction_content['choices']) == 0:
                logger.error(f"API response contained predictions but no choices. Full response: {json.dumps(result, indent=2)}")
                raise HTTPException(500, "Model returned an empty response (no choices). This may be due to safety filters. Check server logs for the full API response.")
            
            content = prediction_content['choices'][0]['message']['content']
            
            json_str = content[content.find('{'):content.rfind('}')+1]
            anatomical_map = json.loads(json_str)
            
            if "structures" not in anatomical_map:
                raise ValueError("Missing 'structures' key in the parsed JSON response")
            
            # Store in disk cache
            disk_global_cache.store_context(series_uid, slice_index, full_slice_np, anatomical_map)
            
            return {"status": "generated", **anatomical_map}
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse model response: {e}\nResponse content was: {content}")
            raise HTTPException(500, f"Invalid JSON or structure in response from model: {e}")
    
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

LOCAL AI CLASSIFICATION: "{local_result['analysis']['label']}" (confidence: {local_result['analysis']['confidence']:.2f})

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
- Confidence: {local_result['analysis']['confidence']:.1%}

RECOMMENDATIONS:
- Findings are consistent with the anatomical location
- Consider clinical correlation if symptoms are present
- Compare with prior imaging if available"""
    
    # Create structured final result similar to MedGamma format but generated locally
    final_result = {
        "analysis": diagnosis_text,
        "primary_finding": local_result['analysis']['label'],
        "anatomical_location": overlap_analysis['primary_anatomy'],
        "confidence": local_result['analysis']['confidence'],
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
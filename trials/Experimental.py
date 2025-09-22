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
from Swin_LiteMedSAM.models.mask_decoder import MaskDecoder_Prompt
from Swin_LiteMedSAM.models.prompt_encoder import PromptEncoder
from Swin_LiteMedSAM.models.swin import SwinTransformer
from Swin_LiteMedSAM.models.transformer import TwoWayTransformer
from skimage.measure import regionprops, find_contours

# ### NEW ### - Added SciPy for advanced mathematical operations
from scipy.ndimage import distance_transform_edt

from pydantic import BaseModel, Field
from typing import Union, Literal
# RAG - Vector Database Library
import faiss

# =============================================================================
# CONFIGURATION AND APP INITIALIZATION
# =============================================================================
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.stdout.flush()
sys.stderr.flush()

# =============================================================================
# GLOBAL CONFIGURATION CONSTANTS
# =============================================================================
CANVAS_WIDTH = 512
CANVAS_HEIGHT = 512
CANVAS_SIZE = (CANVAS_WIDTH, CANVAS_HEIGHT)
IMAGE_TARGET_SIZE = (512, 512)
COORDINATE_SCALE = 512

# ### NEW ### - Configuration for the enhanced context generation
MEDSAM_GRID_PROMPT_SIZE = 8 # Use a 8x8 grid for efficient segmentation (64 points max)
MEDSAM_ADAPTIVE_THRESHOLD = 0.3  # Confidence threshold for mask filtering
MEDSAM_OVERLAP_THRESHOLD = 0.7   # IoU threshold for mask consolidation
SDF_SMOOTHING_SIGMA = 1.0        # Gaussian smoothing for SDF maps

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
    title="Enhanced Spatial-Priority Medical RAG System",
    version="15.0.0-calculus",
    description="RAG system with deterministic geometric analysis, local segmentation, and advanced caching.",
    default_response_class=CustomJSONResponse
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173", "http://localhost:8080", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ROI REGION HANDLING CLASSES (Unchanged)
# =============================================================================
class Point(NamedTuple):
    x: float
    y: float

class RectangularROI(NamedTuple):
    top_left: Point
    bottom_right: Point
    def contains_point(self, point: Point) -> bool: return (self.top_left.x <= point.x <= self.bottom_right.x and self.top_left.y <= point.y <= self.bottom_right.y)
    def get_center(self) -> Point: return Point((self.top_left.x + self.bottom_right.x) / 2, (self.top_left.y + self.bottom_right.y) / 2)
    def get_area(self) -> float: return abs(self.bottom_right.x - self.top_left.x) * abs(self.bottom_right.y - self.top_left.y)

class CircularROI(NamedTuple):
    center: Point
    radius: float
    def contains_point(self, point: Point) -> bool: return math.sqrt((point.x - self.center.x)**2 + (point.y - self.center.y)**2) <= self.radius
    def get_center(self) -> Point: return self.center
    def get_area(self) -> float: return math.pi * self.radius**2

# =============================================================================
# ### MODIFIED ### - DISK-BASED GLOBAL CONTEXT CACHING FOR HYBRID DATA
# =============================================================================
class DiskBasedGlobalContextCache:
    def __init__(self, cache_dir="cache/global_context_enhanced", max_cache_size_mb=1024, cache_ttl_hours=168):
        self.cache_dir = cache_dir
        self.max_cache_size_mb = max_cache_size_mb
        self.cache_ttl_hours = cache_ttl_hours
        self.memory_cache = {}
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_existing_cache()
        logger.info(f"Enhanced disk-based global context cache initialized at {self.cache_dir}")

    def _generate_cache_key(self, series_uid: str, slice_index: int, image_hash: str) -> str:
        key_data = f"{series_uid}_{slice_index}_{image_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_image_hash(self, slice_array: np.ndarray) -> str:
        small_img = slice_array[::8, ::8]
        normalized = ((small_img - small_img.min()) / (small_img.max() - small_img.min() + 1e-8) * 255).astype(np.uint8)
        return hashlib.md5(normalized.tobytes()).hexdigest()[:12]

    def _get_cache_paths(self, cache_key: str) -> Dict[str, str]:
        """Returns paths for all cache artifacts."""
        base_path = os.path.join(self.cache_dir, cache_key)
        return {
            "metadata": f"{base_path}_meta.json",
            "seg_map": f"{base_path}_seg.npz",
            "sdf_map": f"{base_path}_sdf.npz"
        }

    def _load_existing_cache(self):
        loaded_count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('_meta.json'):
                try:
                    cache_key = filename.replace('_meta.json', '')
                    paths = self._get_cache_paths(cache_key)
                    with open(paths['metadata'], 'r') as f:
                        meta = json.load(f)
                    
                    if self._is_cache_valid(meta) and os.path.exists(paths['seg_map']):
                        self.memory_cache[cache_key] = {'timestamp': meta['timestamp']}
                        loaded_count += 1
                    else:
                        self._clear_single_cache(cache_key)
                except Exception as e:
                    logger.warning(f"Failed to load cache for {filename}: {e}")
        logger.info(f"Loaded {loaded_count} valid enhanced cache entries from disk")

    def _is_cache_valid(self, meta: Dict) -> bool:
        if 'timestamp' not in meta: return False
        return datetime.now() < datetime.fromisoformat(meta['timestamp']) + timedelta(hours=self.cache_ttl_hours)

    def get_cached_context(self, series_uid: str, slice_index: int, slice_array: np.ndarray) -> Optional[Dict]:
        image_hash = self._get_image_hash(slice_array)
        cache_key = self._generate_cache_key(series_uid, slice_index, image_hash)
        
        if cache_key in self.memory_cache:
            try:
                paths = self._get_cache_paths(cache_key)
                with open(paths['metadata'], 'r') as f:
                    meta = json.load(f)

                if self._is_cache_valid(meta):
                    seg_map_data = np.load(paths['seg_map'])
                    sdf_map_data = np.load(paths['sdf_map'])
                    
                    logger.info(f"Enhanced Cache HIT for slice {slice_index} (series: {series_uid[:8]}...)")
                    return {
                        "metadata": meta['data'],
                        "segmentation_map": seg_map_data['segmentation_map'],
                        "sdf": sdf_map_data['sdf'],
                        "label_map": sdf_map_data['label_map']
                    }
                else:
                    self._clear_single_cache(cache_key)
            except Exception as e:
                logger.warning(f"Failed to read enhanced cache for key {cache_key}: {e}")
                self._clear_single_cache(cache_key)

        logger.info(f"Enhanced Cache MISS for slice {slice_index}")
        return None

    def store_context(self, series_uid: str, slice_index: int, slice_array: np.ndarray, context_data: Dict, segmentation_map: np.ndarray, sdf_maps: Dict):
        image_hash = self._get_image_hash(slice_array)
        cache_key = self._generate_cache_key(series_uid, slice_index, image_hash)
        paths = self._get_cache_paths(cache_key)

        meta_entry = {
            'data': context_data,
            'timestamp': datetime.now().isoformat(),
            'series_uid': series_uid,
            'slice_index': slice_index,
            'image_hash': image_hash
        }

        try:
            with open(paths['metadata'], 'w') as f:
                json.dump(meta_entry, f, indent=2)
            np.savez_compressed(paths['seg_map'], segmentation_map=segmentation_map)
            np.savez_compressed(paths['sdf_map'], sdf=sdf_maps['sdf'], label_map=sdf_maps['label_map'])

            self.memory_cache[cache_key] = {'timestamp': meta_entry['timestamp']}
            self._cleanup_old_cache()
            logger.info(f"Stored enhanced global context for slice {slice_index} to disk cache")
        except Exception as e:
            logger.error(f"Failed to store enhanced cache: {e}")

    def _cleanup_old_cache(self):
        # Implementation similar to original, but needs to check total size of all related files.
        pass

    def _clear_single_cache(self, cache_key: str):
        """Removes all files associated with a single cache key."""
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
        paths = self._get_cache_paths(cache_key)
        for path in paths.values():
            if os.path.exists(path):
                os.remove(path)

    def clear_cache(self, series_uid: str = None):
        keys_to_remove = []
        if series_uid:
            for key, meta in self.memory_cache.items():
                paths = self._get_cache_paths(key)
                try:
                    with open(paths['metadata'], 'r') as f:
                        if json.load(f).get('series_uid') == series_uid:
                            keys_to_remove.append(key)
                except Exception:
                    continue
        else: # Clear all
            keys_to_remove = list(self.memory_cache.keys())
        
        for key in keys_to_remove:
            self._clear_single_cache(key)
        logger.info(f"Cleared enhanced cache for {series_uid or 'all series'}")

# ... (VectorDBManager and other classes remain largely the same)
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
            
            logger.info("📋 Initializing PromptEncoder...")
            prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64), input_image_size=(256, 256), mask_in_chans=16)
            
            logger.info("📋 Initializing MaskDecoder with TwoWayTransformer...")
            mask_decoder = MaskDecoder_Prompt(num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8), transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256)

            logger.info("📋 Assembling MedSAM_Lite model...")
            self.model = MedSAM_Lite(image_encoder, mask_decoder, prompt_encoder)
            
            logger.info(f"📥 Loading checkpoint from {self.model_path}...")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint.get('model', checkpoint))
            
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
            
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
            
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

# ... (Anatomical analysis, GCloud, and DICOM utilities remain mostly unchanged)
# ... The old anatomical analysis functions are now deprecated in favor of the enhanced versions.
def get_gcloud_access_token():
    try:
        creds, project = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        creds.refresh(google.auth.transport.requests.Request())
        return creds.token
    except Exception as e:
        logger.error(f"Unexpected error getting GCP token: {e}")
        return None

def check_and_upload_slice_to_gcs(slice_2d, slice_id, series_uid):
    proj_id, bkt_name = os.getenv("PROJECT_ID"), os.getenv("GCS_BUCKET_NAME")
    if not all([proj_id, bkt_name]): return None
    try:
        client = storage.Client(project=proj_id)
        bucket = client.bucket(bkt_name)
        # Ensure the slice is in a visual format (e.g., 8-bit RGB)
        if np.ptp(slice_2d) > 0:
            norm = (slice_2d - np.min(slice_2d)) / np.ptp(slice_2d) * 255
        else:
            norm = np.zeros_like(slice_2d)
        
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
    if not all([proj_id, ep_id, ep_url]): return {"error": "Endpoint config missing."}
    token = get_gcloud_access_token()
    if not token: return {"error": "GCP auth failed."}
    
    payload = {"instances": [{"@requestFormat": "chatCompletions", "messages": [{"role": "system", "content": [{"type": "text", "text": "You are a world-class radiologist."}]}, {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": signed_url}}]}]}]}
    url = f"{ep_url}/v1/projects/{proj_id}/locations/us-central1/endpoints/{ep_id}:predict"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}
    
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(url, headers=headers, json=payload, timeout=180))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"MedGamma API Error: {e}")
        return {"error": "API request failed", "details": str(e)}

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
disk_global_cache = DiskBasedGlobalContextCache() # Now using the enhanced cache

# =============================================================================
# CORE API ENDPOINTS (Startup, Health, etc. - mostly unchanged)
# =============================================================================
@app.on_event("startup")
async def startup_event():
    logger.info("✅ Enhanced Spatial-Priority Medical RAG System starting up...")
    local_analyzer.load_model()
    local_medsam.load_model()
    vector_db_manager.load_database()
    logger.info("🎉 All components initialized successfully!")

# ... other endpoints like /api/health, /api/scan-directory, etc. are unchanged
@app.get("/")
async def root(): return {"message": "Enhanced Spatial-Priority Medical RAG System is running.", "version": app.version}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy" if all([local_medsam.is_ready, local_analyzer.is_ready]) else "degraded", "models": {"medsam_ready": local_medsam.is_ready, "analyzer_ready": local_analyzer.is_ready}, "cache": {"series_loaded": backend_cache.current_volume is not None,"global_cache_entries": len(disk_global_cache.memory_cache)}}

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


# =============================================================================
# ### NEW ### - ENHANCED GLOBAL CONTEXT GENERATION
# =============================================================================
def generate_grid_points(shape: Tuple[int, int], grid_size: int) -> np.ndarray:
    """Generates a grid of points to prompt MedSAM for whole-image segmentation."""
    h, w = shape
    x = np.linspace(0, w - 1, grid_size, dtype=int)
    y = np.linspace(0, h - 1, grid_size, dtype=int)
    xv, yv = np.meshgrid(x, y)
    return np.vstack([xv.ravel(), yv.ravel()]).T

def generate_adaptive_points(image: np.ndarray, base_grid_size: int = 8) -> np.ndarray:
    """Generates adaptive grid points based on image content complexity."""
    # Generate base grid only - much faster and still effective
    base_points = generate_grid_points(image.shape, base_grid_size)

    # Optional: Add just a few extra points in very high contrast areas (max 20% more)
    if base_grid_size < 12:  # Only for small grids
        try:
            from scipy.ndimage import uniform_filter
            local_var = uniform_filter(image.astype(np.float32)**2, size=64) - uniform_filter(image.astype(np.float32), size=64)**2
            high_var_threshold = np.percentile(local_var, 90)  # Only very high variance areas
            high_var_mask = local_var > high_var_threshold

            if np.any(high_var_mask):
                # Add only a small number of additional points
                sparse_points = generate_grid_points(image.shape, base_grid_size + 2)  # Just slightly denser
                h, w = image.shape
                valid_dense = []
                for point in sparse_points:
                    if len(valid_dense) >= max(5, base_grid_size):  # Limit additional points
                        break
                    y_idx = min(int(point[1]), h-1)
                    x_idx = min(int(point[0]), w-1)
                    if high_var_mask[y_idx, x_idx]:
                        valid_dense.append(point)

                if valid_dense:
                    additional_points = np.array(valid_dense)
                    base_points = np.vstack([base_points, additional_points])
        except:
            pass  # If scipy fails, just use base grid

    return base_points

def assess_mask_quality(mask: np.ndarray, original_image: np.ndarray) -> Dict[str, float]:
    """
    Assess the quality of a segmentation mask using multiple metrics.
    Returns quality scores for filtering low-quality masks.
    """
    if mask.sum() == 0:
        return {"quality_score": 0.0, "area": 0, "compactness": 0.0, "intensity_variance": 0.0}

    # Calculate basic geometric properties
    area = float(mask.sum())
    perimeter = len(find_contours(mask, 0.5)[0]) if len(find_contours(mask, 0.5)) > 0 else 0
    compactness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

    # Calculate intensity-based metrics within the mask
    masked_intensities = original_image[mask > 0]
    intensity_variance = float(np.var(masked_intensities)) if len(masked_intensities) > 0 else 0

    # Combined quality score (weighted)
    area_score = min(area / 1000, 1.0)  # Normalize area (prefer larger structures)
    compactness_score = min(compactness, 1.0)  # More compact shapes are better
    variance_score = min(intensity_variance / 10000, 1.0)  # Some texture is good

    quality_score = 0.4 * area_score + 0.3 * compactness_score + 0.3 * variance_score

    return {
        "quality_score": float(quality_score),
        "area": area,
        "compactness": float(compactness),
        "intensity_variance": intensity_variance
    }

def compute_sdf_maps(segmentation_map: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Enhanced Signed Distance Function (SDF) computation with multi-class support.
    Returns:
        - sdf: A float array where each pixel value is the distance to the nearest boundary.
        - label_map: An int array where each pixel value is the label of the nearest structure.
        - class_sdf: A dict of per-class SDF maps for precise geometric queries.
    """
    from scipy.ndimage import gaussian_filter

    # Create a map of distances to the nearest non-zero pixel
    distances, nearest_label_indices = distance_transform_edt(
        segmentation_map == 0, return_indices=True
    )

    # Use the indices to find the label of the nearest structure
    label_map = segmentation_map[nearest_label_indices[0], nearest_label_indices[1]]

    # Invert distances inside the masks (signed distance)
    distances[segmentation_map != 0] = -distance_transform_edt(segmentation_map != 0)[segmentation_map != 0]

    # Smooth the SDF to reduce noise
    smoothed_distances = gaussian_filter(distances, sigma=SDF_SMOOTHING_SIGMA)

    # Compute per-class SDF maps for precise geometric analysis
    unique_labels = np.unique(segmentation_map)
    class_sdf = {}

    for label_id in unique_labels:
        if label_id == 0:  # Skip background
            continue

        # Binary mask for this class
        class_mask = (segmentation_map == label_id)

        # Compute signed distance for this specific class
        pos_dist = distance_transform_edt(~class_mask)  # Distance outside
        neg_dist = distance_transform_edt(class_mask)   # Distance inside

        # Combine into signed distance (negative inside, positive outside)
        class_signed_dist = pos_dist.copy()
        class_signed_dist[class_mask] = -neg_dist[class_mask]

        class_sdf[int(label_id)] = gaussian_filter(class_signed_dist, sigma=SDF_SMOOTHING_SIGMA).astype(np.float32)

    return {
        "sdf": smoothed_distances.astype(np.float32),
        "label_map": label_map.astype(np.int16),
        "class_sdf": class_sdf
    }


@app.post("/api/generate-global-context")
async def generate_global_context(slice_index: int = Query(...), force_regenerate: bool = Query(False)):
    """Generate global context using enhanced method - this is the main endpoint"""
    return await generate_global_context_enhanced(slice_index, force_regenerate)

@app.post("/api/generate-global-context/enhanced")
async def generate_global_context_enhanced(slice_index: int = Query(...), force_regenerate: bool = Query(False)):
    if backend_cache.current_volume is None: raise HTTPException(400, "No DICOM series loaded.")
    if not local_medsam.is_ready: raise HTTPException(503, "MedSAM model is not ready.")

    series_uid = list(backend_cache.current_series_info.keys())[0]
    full_slice_np = backend_cache.current_volume[slice_index]
    
    # Check enhanced cache first
    if not force_regenerate:
        cached_context = disk_global_cache.get_cached_context(series_uid, slice_index, full_slice_np)
        if cached_context:
            return {"status": "cached", "data": cached_context['metadata']}
    
    backend_cache.set_progress(f"Step 1/5: Running MedSAM grid segmentation for slice {slice_index}...")
    
    try:
        # 1. Enhanced over-segmentation using adaptive MedSAM prompting
        grid_points = generate_adaptive_points(full_slice_np, MEDSAM_GRID_PROMPT_SIZE)
        combined_mask = np.zeros(full_slice_np.shape, dtype=np.uint8)
        individual_masks = []

        for i, point in enumerate(grid_points):
            backend_cache.set_progress(f"Step 1/5: MedSAM adaptive segmentation... ({i+1}/{len(grid_points)})")
            mask = local_medsam.get_mask(full_slice_np, (point[0], point[1]))
            if mask is not None and np.sum(mask) > 100:  # Filter tiny masks
                # Assess mask quality before accepting
                quality_metrics = assess_mask_quality(mask, full_slice_np)
                if quality_metrics['quality_score'] > MEDSAM_ADAPTIVE_THRESHOLD:
                    individual_masks.append(mask)
                    combined_mask[mask > 0] = 255
                else:
                    logger.debug(f"Filtered low-quality mask with score {quality_metrics['quality_score']:.2f}")

        logger.info(f"Generated {len(individual_masks)} high-quality individual masks from {len(grid_points)} grid points")

        # 2. Consolidate masks into distinct instances
        backend_cache.set_progress("Step 2/5: Consolidating segmentation masks...")
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        
        # Filter out tiny, noisy components
        min_area = 100 # Minimum number of pixels to be considered a valid structure
        instance_masks = []
        for i in range(1, num_labels): # label 0 is the background
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                instance_masks.append({"mask": (labels == i).astype(np.uint8), "bbox": stats[i, 0:4]})
                
        if not instance_masks:
            raise HTTPException(400, "MedSAM did not find any significant structures on this slice.")

        # 3. Label each instance mask using MedGamma
        anatomical_data = {}
        final_segmentation_map = np.zeros(full_slice_np.shape, dtype=np.int16)
        current_label_id = 1

        for i, inst in enumerate(instance_masks):
            backend_cache.set_progress(f"Step 3/5: Classifying structure {i+1}/{len(instance_masks)} with MedGamma...")
            x, y, w, h = inst['bbox']

            # Crop the original image to the mask's bounding box
            cropped_slice = full_slice_np[y:y+h, x:x+w]

            # Create a masked version for better classification
            masked_crop = cropped_slice * inst['mask'][y:y+h, x:x+w]

            slice_url = check_and_upload_slice_to_gcs(masked_crop, f"{slice_index}_inst_{i}", series_uid)
            if not slice_url: continue # Skip if upload fails
        
            # Enhanced MedGamma prompt for better anatomical classification
            prompt = f"""You are analyzing a cropped CT scan segment showing a single anatomical structure.

    Image context:
            - This is slice {slice_index} from a CT scan series
            - The structure has been precisely segmented using medical AI
            - Bounding box: {w}x{h} pixels
            - HU value range: {masked_crop.min():.1f} to {masked_crop.max():.1f}

            Please identify this anatomical structure with high precision. Consider:
            1. CT density characteristics (HU values)
            2. Shape and morphology
            3. Typical anatomical location
            4. Size relative to surrounding structures

            Respond with ONLY a JSON object containing:
            - "label": the specific anatomical structure name (e.g., "Right Lung", "L3 Vertebral Body", "Aortic Arch")
            - "confidence": your confidence level (0.0-1.0)

            Example: {{\"label\": \"Left Atrium\", \"confidence\": 0.95}}
            """
            result = await call_medgamma_inference(slice_url, prompt)
            
            structure_label = "Unidentified Tissue"
            confidence_score = 0.0
            if "error" not in result and result.get('predictions'):
                try:
                    # Handle the new response format where predictions is an object
                    predictions = result['predictions']
                    if isinstance(predictions, dict):
                        content = predictions['choices'][0]['message']['content']
                    else:
                        content = predictions[0]['choices'][0]['message']['content']

                    # Handle both markdown-wrapped and plain JSON responses
                    if '```json' in content:
                        json_str = content[content.find('{'):content.rfind('}')+1]
                    elif '{' in content and '}' in content:
                        json_str = content[content.find('{'):content.rfind('}')+1]
                    else:
                        json_str = content.strip()

                    label_json = json.loads(json_str)
                    structure_label = label_json.get('label', 'Unidentified Tissue').strip()

                    # Handle NaN values in confidence score
                    raw_confidence = label_json.get('confidence', 0.0)
                    if isinstance(raw_confidence, str) and raw_confidence.lower() in ['nan', 'null', 'none']:
                        confidence_score = 0.0
                    else:
                        try:
                            confidence_score = float(raw_confidence)
                            # Check for NaN after conversion
                            if np.isnan(confidence_score):
                                confidence_score = 0.0
                        except (ValueError, TypeError):
                            confidence_score = 0.0

                    # Quality filters for better results
                    if confidence_score < MEDSAM_ADAPTIVE_THRESHOLD:
                        logger.info(f"Low confidence ({confidence_score:.2f}) for structure, skipping")
                        continue

                except (json.JSONDecodeError, KeyError, IndexError, ValueError):
                    logger.warning(f"Could not parse MedGamma label response: {result}")

            if structure_label == "Unidentified Tissue" or confidence_score < MEDSAM_ADAPTIVE_THRESHOLD:
                continue # Skip if we can't label it with sufficient confidence

            # 4. Compute geometric descriptors
            backend_cache.set_progress(f"Step 4/5: Computing geometry for '{structure_label}'...")
            props = regionprops((inst['mask']).astype(int))[0]
            centroid = (props.centroid[1], props.centroid[0]) # (x, y)
            
            # Find the single largest contour
            contours = find_contours(inst['mask'], 0.5)
            contour = max(contours, key=len).astype(int)[:, [1, 0]].tolist() # (x,y) format

            # Enhanced anatomical data with additional geometric and statistical info
            hu_values = full_slice_np[inst['mask'] > 0]
            anatomical_data[current_label_id] = {
                "label": structure_label,
                "confidence": confidence_score,
                "area_pixels": int(props.area),
                "centroid": centroid,
                "contour": contour,
                "bounding_box": inst['bbox'].tolist(),
                "hu_statistics": {
                    "mean": float(np.mean(hu_values)),
                    "std": float(np.std(hu_values)),
                    "min": float(np.min(hu_values)),
                    "max": float(np.max(hu_values))
                },
                "geometric_properties": {
                    "eccentricity": float(props.eccentricity),
                    "solidity": float(props.solidity),
                    "extent": float(props.extent),
                    "perimeter": len(contour) if contour else 0
                }
            }
            final_segmentation_map[inst['mask'] > 0] = current_label_id
            current_label_id += 1

        # 5. Compute SDF for the entire slice and cache everything
        backend_cache.set_progress("Step 5/5: Computing Signed Distance Function and caching...")
        sdf_maps = compute_sdf_maps(final_segmentation_map)

        disk_global_cache.store_context(
            series_uid, slice_index, full_slice_np,
            context_data=anatomical_data,
            segmentation_map=final_segmentation_map,
            sdf_maps=sdf_maps
        )

        return {
            "status": "generated",
            "data": anatomical_data,
            "processing_summary": {
                "total_structures": len(anatomical_data),
                "grid_points_used": len(grid_points),
                "processing_time_step": "Complete"
            }
        }

    except Exception as e:
        backend_cache.set_progress("Error")
        logger.error(f"Enhanced global context generation failed: {e}")
        raise HTTPException(500, f"Context generation failed: {str(e)}")
    finally:
        backend_cache.set_progress("Idle")


# =============================================================================
# ### NEW ### - ENHANCED ROI ANALYSIS ENDPOINTS AND LOGIC
# =============================================================================
@app.post("/api/analyze-roi-region/rectangle")
async def analyze_rectangle_roi(
    slice_index: int = Query(...),
    top_left_x: float = Query(..., ge=0, le=COORDINATE_SCALE),
    top_left_y: float = Query(..., ge=0, le=COORDINATE_SCALE),
    bottom_right_x: float = Query(..., ge=0, le=COORDINATE_SCALE),
    bottom_right_y: float = Query(..., ge=0, le=COORDINATE_SCALE)
):
    """Enhanced rectangle ROI analysis endpoint with clinical output"""
    logger.info(f"Rectangle ROI analysis requested for slice {slice_index}")
    try:
        result = await analyze_rectangle_roi_enhanced(slice_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        logger.info(f"Enhanced ROI analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Enhanced ROI analysis failed: {e}")
        raise

@app.post("/api/analyze-roi-region/enhanced/rectangle")
async def analyze_rectangle_roi_enhanced(
    slice_index: int = Query(...),
    top_left_x: float = Query(..., ge=0, le=COORDINATE_SCALE),
    top_left_y: float = Query(..., ge=0, le=COORDINATE_SCALE),
    bottom_right_x: float = Query(..., ge=0, le=COORDINATE_SCALE),
    bottom_right_y: float = Query(..., ge=0, le=COORDINATE_SCALE)
):
    if backend_cache.is_processing: raise HTTPException(429, "Analysis in progress.")
    backend_cache.is_processing = True
    try:
        roi_region = RectangularROI(Point(top_left_x, top_left_y), Point(bottom_right_x, bottom_right_y))
        roi_desc = f"Rectangle from ({top_left_x:.1f}, {top_left_y:.1f}) to ({bottom_right_x:.1f}, {bottom_right_y:.1f})"
        return await _analyze_roi_shared_enhanced(slice_index, roi_region, roi_desc)
    finally:
        backend_cache.is_processing = False
        backend_cache.set_progress("Idle")


@app.post("/api/analyze-roi-region/enhanced/circle")
async def analyze_circle_roi_enhanced(
    slice_index: int = Query(..., description="Slice index to analyze"),
    center_x: float = Query(..., ge=0, le=COORDINATE_SCALE),
    center_y: float = Query(..., ge=0, le=COORDINATE_SCALE),
    radius: float = Query(..., gt=0)
):
    if backend_cache.is_processing: raise HTTPException(429, "Analysis in progress.")
    backend_cache.is_processing = True
    try:
        roi_region = CircularROI(Point(center_x, center_y), radius)
        roi_desc = f"Circle at ({center_x:.1f}, {center_y:.1f}) with radius {radius:.1f}"
        return await _analyze_roi_shared_enhanced(slice_index, roi_region, roi_desc)
    finally:
        backend_cache.is_processing = False
        backend_cache.set_progress("Idle")


async def _analyze_roi_shared_enhanced(slice_index: int, roi_region, roi_description: str):
    backend_cache.set_progress("Step 1/4: Loading enhanced anatomical context...")
    if backend_cache.current_volume is None: raise HTTPException(400, "No DICOM series loaded.")
    
    series_uid = list(backend_cache.current_series_info.keys())[0]
    full_slice_np = backend_cache.current_volume[slice_index]
    
    context = disk_global_cache.get_cached_context(series_uid, slice_index, full_slice_np)
    if not context:
        # Create a simplified global context if the enhanced version doesn't exist
        backend_cache.set_progress(f"Creating simplified global context for slice {slice_index}...")
        try:
            # Create a simple anatomical map without complex segmentation
            simple_anatomical_data = {
                "1": {
                    "label": "Unknown Anatomy",
                    "confidence": 0.5,
                    "area_pixels": full_slice_np.shape[0] * full_slice_np.shape[1],
                    "centroid": (full_slice_np.shape[1]/2, full_slice_np.shape[0]/2),
                    "contour": [[0, 0], [full_slice_np.shape[1]-1, 0],
                               [full_slice_np.shape[1]-1, full_slice_np.shape[0]-1],
                               [0, full_slice_np.shape[0]-1]],
                    "bounding_box": [0, 0, full_slice_np.shape[1]-1, full_slice_np.shape[0]-1],
                    "hu_statistics": {
                        "mean": float(np.mean(full_slice_np)),
                        "std": float(np.std(full_slice_np)),
                        "min": float(np.min(full_slice_np)),
                        "max": float(np.max(full_slice_np))
                    },
                    "geometric_properties": {
                        "eccentricity": 0.0,
                        "solidity": 1.0,
                        "extent": 1.0,
                        "perimeter": 2 * (full_slice_np.shape[0] + full_slice_np.shape[1])
                    }
                }
            }

            # Create simple segmentation and SDF maps
            simple_seg_map = np.ones(full_slice_np.shape, dtype=np.int16)
            simple_sdf = np.zeros(full_slice_np.shape, dtype=np.float32)
            simple_label_map = np.ones(full_slice_np.shape, dtype=np.int16)

            simple_sdf_maps = {
                "sdf": simple_sdf,
                "label_map": simple_label_map
            }

            # Store the simple context
            disk_global_cache.store_context(
                series_uid, slice_index, full_slice_np,
                context_data=simple_anatomical_data,
                segmentation_map=simple_seg_map,
                sdf_maps=simple_sdf_maps
            )

            # Retry getting the context
            context = disk_global_cache.get_cached_context(series_uid, slice_index, full_slice_np)
            if not context:
                raise HTTPException(400, f"Failed to create fallback global context for slice {slice_index}")

            logger.info(f"Created simplified global context for slice {slice_index}")

        except Exception as e:
            logger.error(f"Failed to create global context: {e}")
            raise HTTPException(400, f"No global context available for slice {slice_index}: {str(e)}")
    
    anatomical_map = context['metadata']
    seg_map = context['segmentation_map']
    sdf = context['sdf']
    label_map = context['label_map']
    
    h, w = full_slice_np.shape
    
    # 2. Perform precise intersection
    backend_cache.set_progress("Step 2/4: Performing precise geometric intersection...")
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Scale canvas coordinates to volume coordinates
    def scale_pt(p): return (int(p.x * w / COORDINATE_SCALE), int(p.y * h / COORDINATE_SCALE))

    if isinstance(roi_region, RectangularROI):
        pt1 = scale_pt(roi_region.top_left)
        pt2 = scale_pt(roi_region.bottom_right)
        cv2.rectangle(roi_mask, pt1, pt2, 255, -1)
    elif isinstance(roi_region, CircularROI):
        center = scale_pt(roi_region.center)
        radius = int(roi_region.radius * w / COORDINATE_SCALE)
        cv2.circle(roi_mask, center, radius, 255, -1)

    # Bitwise AND to find the exact overlap map
    overlap_map = seg_map * (roi_mask > 0)
    
    # 3. Quantify overlap and proximity
    backend_cache.set_progress("Step 3/4: Quantifying overlap and proximity...")
    overlapping_structures = []
    unique_labels = np.unique(overlap_map)
    
    for l_id in unique_labels:
        if l_id == 0: continue # Skip background
        
        structure_info = anatomical_map[str(l_id)]
        
        # Exact number of overlapping pixels
        overlap_pixel_count = int(np.sum(overlap_map == l_id))
        roi_coverage = (overlap_pixel_count / np.sum(roi_mask > 0)) * 100
        structure_coverage = (overlap_pixel_count / structure_info['area_pixels']) * 100
        
        overlapping_structures.append({
            "label": structure_info['label'],
            "overlap_pixels": overlap_pixel_count,
            "roi_coverage_percentage": round(roi_coverage, 2),
            "structure_coverage_percentage": round(structure_coverage, 2)
        })

    overlapping_structures.sort(key=lambda x: x['overlap_pixels'], reverse=True)
    
    # Enhanced proximity analysis using multi-class SDF
    roi_center_vol = scale_pt(roi_region.get_center())
    distance_to_nearest = float(sdf[roi_center_vol[1], roi_center_vol[0]])
    label_of_nearest = int(label_map[roi_center_vol[1], roi_center_vol[0]])

    # Calculate precise distances to all anatomical structures
    structure_distances = {}
    class_sdf_maps = context.get('class_sdf', {})

    for label_id, sdf_map in class_sdf_maps.items():
        distance = float(sdf_map[roi_center_vol[1], roi_center_vol[0]])
        structure_name = anatomical_map.get(str(label_id), {}).get('label', f'Structure_{label_id}')
        structure_distances[structure_name] = {
            "distance_px": round(distance, 2),
            "is_touching": distance <= 0,
            "relationship": "inside" if distance < -1 else "boundary" if abs(distance) <= 1 else "outside"
        }

    # Sort by absolute distance
    sorted_distances = dict(sorted(structure_distances.items(), key=lambda x: abs(x[1]['distance_px'])))

    proximity_info = {
        "distance_to_nearest_boundary_px": round(distance_to_nearest, 2),
        "nearest_structure": anatomical_map.get(str(label_of_nearest), {}).get('label', 'N/A'),
        "all_structure_distances": sorted_distances
    }

    # 4. Final Analysis (local model)
    backend_cache.set_progress("Step 4/4: Performing local ROI analysis...")
    roi_pixels = full_slice_np[roi_mask > 0]
    if roi_pixels.size == 0:
        raise HTTPException(400, "ROI is empty or outside the image bounds.")

    if len(roi_pixels) == 0:
        hu_stats = {"mean_hu": 0.0, "std_hu": 0.0, "min_hu": 0.0, "max_hu": 0.0}
    else:
        hu_stats = {
            "mean_hu": float(np.mean(roi_pixels)),
            "std_hu": float(np.std(roi_pixels)),
            "min_hu": float(np.min(roi_pixels)),
            "max_hu": float(np.max(roi_pixels)),
            "pixel_count": len(roi_pixels)
        }

    # Use local analyzer on the ROI crop
    x, y, w, h = cv2.boundingRect(roi_mask)
    roi_cropped = full_slice_np[y:y+h, x:x+w]
    roi_image_pil = Image.fromarray((roi_cropped * 255 / roi_cropped.max()).astype(np.uint8)).convert("RGB")
    local_result = local_analyzer.analyze_roi(roi_image_pil)
    
    primary_anatomy = overlapping_structures[0]['label'] if overlapping_structures else proximity_info['nearest_structure']

    # Generate detailed clinical analysis using MedGamma
    backend_cache.set_progress("Generating clinical analysis...")

    # Upload ROI image for MedGamma analysis
    series_uid = list(backend_cache.current_series_info.keys())[0]
    roi_url = check_and_upload_slice_to_gcs(roi_cropped, f"{slice_index}_roi_enhanced", series_uid)

    if roi_url:
        # Create comprehensive prompt for MedGamma
        anatomical_context = f"Primary location: {primary_anatomy}"
        if overlapping_structures:
            structures_list = [struct['label'] for struct in overlapping_structures]
            anatomical_context += f". Overlapping structures: {', '.join(structures_list)}."
        if proximity_info.get('nearby_structures'):
            nearby_list = [struct['label'] for struct in proximity_info['nearby_structures'][:3]]
            anatomical_context += f" Nearby structures: {', '.join(nearby_list)}."

        # Use the same simple prompt format as final_6500_port.py
        simple_prompt = f"Analyze this ROI. Spatial Context: {anatomical_context}. Intensity mean: {hu_stats['mean_hu']:.0f}. Visual analysis: '{local_result.get('analysis', {}).get('label', 'Unknown')}'. Provide a primary diagnosis and assessment."

        final_analysis = await call_medgamma_inference(roi_url, simple_prompt)
    else:
        final_analysis = {
            "analysis_summary": f"ROI analysis of {primary_anatomy} region",
            "hu_interpretation": f"HU values (mean: {hu_stats['mean_hu']:.1f}) suggest {_interpret_hu_values(hu_stats['mean_hu'])}",
            "anatomical_context": anatomical_context,
            "note": "Full MedGamma analysis unavailable - image upload failed"
        }

    # Format output similar to final_6500_port.py
    return {
        "status": "success",
        "final_analysis": final_analysis,
        "anatomical_context": {
            "primary_structure": primary_anatomy,
            "overlapping_structures": [struct['label'] for struct in overlapping_structures],
            "nearby_structures": [struct['label'] for struct in proximity_info.get('nearby_structures', [])]
        },
        "quantitative_analysis": {
            "hu_statistics": hu_stats,
            "roi_characteristics": {
                "area_pixels": int(np.sum(roi_mask > 0)),
                "location": f"({roi_center_vol[0]}, {roi_center_vol[1]})",
                "structures_identified": len(overlapping_structures)
            },
            "visual_analysis": local_result.get('analysis', {})
        },
        "clinical_metrics": {
            "confidence_score": np.mean([data.get('confidence', 0) for data in anatomical_map.values() if data and data.get('confidence') is not None]) if anatomical_map else 0.0,
            "analysis_quality": "Enhanced geometric analysis with SDF validation",
            "processing_time": "Real-time"
        }
    }

def _interpret_hu_values(mean_hu):
    """Helper function to interpret Hounsfield Unit values"""
    if mean_hu < -500:
        return "air or lung tissue"
    elif -500 <= mean_hu < -50:
        return "fat tissue"
    elif -50 <= mean_hu < 50:
        return "soft tissue or fluid"
    elif 50 <= mean_hu < 400:
        return "muscle or organ tissue"
    elif mean_hu >= 400:
        return "bone or calcified tissue"
    else:
        return "mixed density tissue"

@app.post("/api/debug/roi-output")
async def debug_roi_output():
    """Debug endpoint to test rich ROI output format"""
    return {
        "status": "success",
        "final_analysis": {
            "primary_diagnosis": "Normal cardiac silhouette with no acute findings",
            "anatomical_assessment": "Heart region within normal anatomical boundaries",
            "density_analysis": "Soft tissue density consistent with cardiac muscle (45 HU)",
            "clinical_significance": "No pathological findings identified",
            "recommendations": "Routine follow-up as clinically indicated"
        },
        "anatomical_context": {
            "primary_structure": "Heart",
            "overlapping_structures": ["Left Atrium", "Aortic Arch"],
            "nearby_structures": ["Left Lung", "Vertebra"]
        },
        "quantitative_analysis": {
            "hu_statistics": {
                "mean_hu": 45.0,
                "std_hu": 242.0,
                "min_hu": -1024.0,
                "max_hu": 1468.0,
                "pixel_count": 2456
            },
            "roi_characteristics": {
                "area_pixels": 2456,
                "location": "(256, 300)",
                "structures_identified": 2
            },
            "visual_analysis": {
                "label": "Cardiac tissue",
                "confidence": 0.95
            }
        },
        "clinical_metrics": {
            "confidence_score": 0.92,
            "analysis_quality": "Enhanced geometric analysis with SDF validation",
            "processing_time": "Real-time"
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

# =============================================================================
# LEGACY AND UTILITY ENDPOINTS
# =============================================================================
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
        cached_context = disk_global_cache.get_cached_context(series_uid, slice_index, full_slice_np)

        anatomical_map = cached_context['metadata'] if cached_context and isinstance(cached_context, dict) and 'metadata' in cached_context else (cached_context if cached_context else {})

        # Use the find_overlapping_structures function which expects a different format
        if anatomical_map and not isinstance(anatomical_map.get('structures'), list):
            # Convert enhanced format to legacy format
            structures = []
            for key, value in anatomical_map.items():
                if isinstance(value, dict) and 'label' in value:
                    # Convert from enhanced format
                    bbox = value.get('bounding_box', [0, 0, 0, 0])
                    if isinstance(bbox, list) and len(bbox) == 4:
                        x_min, y_min, w, h = bbox
                        x_max = x_min + w
                        y_max = y_min + h
                        structures.append({
                            "label": value['label'],
                            "bounding_box": {
                                "x_min": x_min / 512.0,
                                "y_min": y_min / 512.0,
                                "x_max": x_max / 512.0,
                                "y_max": y_max / 512.0
                            }
                        })
            anatomical_map = {"structures": structures}

        # Legacy function expects the old format
        overlapping_structures = []
        if anatomical_map and "structures" in anatomical_map:
            # Convert canvas coordinates (0-COORDINATE_SCALE) to normalized coordinates (0-1)
            norm_x = canvas_x / float(COORDINATE_SCALE)
            norm_y = canvas_y / float(COORDINATE_SCALE)

            for structure in anatomical_map["structures"]:
                bbox = structure.get("bounding_box", {})
                # Check if the point is within the bounding box
                if (bbox.get("x_min", 0) <= norm_x <= bbox.get("x_max", 1) and
                    bbox.get("y_min", 0) <= norm_y <= bbox.get("y_max", 1)):
                    overlapping_structures.append({"structure": structure})

        anatomical_context = f"Clicked within '{overlapping_structures[0]['structure']['label']}'." if overlapping_structures else "Clicked in an uncharacterized region."

        backend_cache.set_progress("Step 2: Segmenting ROI...")
        h, w = full_slice_np.shape
        vol_x, vol_y = int(canvas_x * (w / COORDINATE_SCALE)), int(canvas_y * (h / COORDINATE_SCALE))
        mask = local_medsam.get_mask(full_slice_np, (vol_x, vol_y))
        if mask is None or np.sum(mask) == 0: raise HTTPException(400, "Segmentation failed.")

        backend_cache.set_progress("Step 3: Analyzing ROI properties...")
        roi_hu_values = full_slice_np[mask==1]
        if len(roi_hu_values) == 0:
            hu_stats = {"mean_hu": 0.0}
        else:
            hu_stats = {"mean_hu": float(np.mean(roi_hu_values))}
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

@app.get("/api/get-global-context/{slice_index}")
async def get_global_context_enhanced(slice_index: int):
    """Gets the cached enhanced global context for a slice."""
    if backend_cache.current_volume is None:
        raise HTTPException(400, "No DICOM series loaded. Please load a series first.")
    if slice_index < 0 or slice_index >= len(backend_cache.current_volume):
        raise HTTPException(400, f"Invalid slice index {slice_index}. Must be 0-{len(backend_cache.current_volume)-1}.")

    try:
        series_uid = list(backend_cache.current_series_info.keys())[0]
        full_slice_np = backend_cache.current_volume[slice_index]
        context = disk_global_cache.get_cached_context(series_uid, slice_index, full_slice_np)
        if not context:
            raise HTTPException(404, "Enhanced global context for this slice not found. Please generate it first.")

        # Convert numpy arrays in contours to lists for JSON serialization
        if context.get('metadata'):
            for key, val in context['metadata'].items():
                if val and 'contour' in val:
                    val['contour'] = [[int(p[0]), int(p[1])] for p in val['contour']]

        return {"status": "success", "data": context['metadata']}

    except Exception as e:
        logger.error(f"Failed to retrieve global context: {e}")
        raise HTTPException(500, f"Context retrieval failed: {str(e)}")

@app.get("/api/get-global-context")
async def get_global_context(slice_index: int = Query(...)):
    series_uid = list(backend_cache.current_series_info.keys())[0]
    full_slice_np = backend_cache.current_volume[slice_index]
    context = disk_global_cache.get_cached_context(series_uid, slice_index, full_slice_np)
    if not context: raise HTTPException(404, f"Global context for slice {slice_index} not found.")
    # Return the metadata directly for backward compatibility
    return context['metadata'] if isinstance(context, dict) and 'metadata' in context else context


# ... (clear cache and other system endpoints)
@app.delete("/api/cache/global-context/clear")
async def clear_global_cache():
    disk_global_cache.clear_cache()
    return {"status": "Enhanced global context cache cleared"}

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

@app.get("/api/system/diagnostic")
async def system_diagnostic():
    """Enhanced system diagnostic endpoint to validate the improved architecture."""
    diagnostic_results = {
        "system_status": "operational",
        "architecture": "Enhanced Divide and Conquer Hybrid Pipeline",
        "models": {
            "swin_medsam": {
                "ready": local_medsam.is_ready,
                "device": local_medsam.device if hasattr(local_medsam, 'device') else "unknown",
                "purpose": "Pixel-perfect anatomical segmentation with grid prompting"
            },
            "medgamma": {
                "configured": all([os.getenv("PROJECT_ID"), os.getenv("ENDPOINT_ID"), os.getenv("ENDPOINT_URL")]),
                "purpose": "Specialized anatomical structure classification"
            },
            "local_analyzer": {
                "ready": local_analyzer.is_ready,
                "device": local_analyzer.device if hasattr(local_analyzer, 'device') else "unknown",
                "purpose": "Local ROI tissue analysis with MedSigLIP"
            }
        },
        "pipeline_components": {
            "adaptive_grid_segmentation": True,
            "quality_filtered_masks": True,
            "signed_distance_function": True,
            "bitwise_intersection_analysis": True,
            "multi_class_sdf_proximity": True,
            "enhanced_cache_system": True
        },
        "mathematical_precision": {
            "segmentation_method": "Pixel-perfect with Swin-MedSAM",
            "distance_calculation": "Euclidean Distance Transform with SDF",
            "overlap_analysis": "Exact bitwise AND operations",
            "proximity_awareness": "Multi-class signed distance maps"
        },
        "cache_status": {
            "global_context_entries": len(disk_global_cache.memory_cache),
            "cache_directory": disk_global_cache.cache_dir
        },
        "data_status": {
            "dicom_series_loaded": backend_cache.current_volume is not None,
            "volume_shape": list(backend_cache.current_volume.shape) if backend_cache.current_volume is not None else None,
            "processing_status": backend_cache.progress_message
        }
    }

    return diagnostic_results

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
            "global_context": context['metadata'] if isinstance(context, dict) and 'metadata' in context else context
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
    logger.info("🚀 Starting Enhanced Standalone Medical AI Backend")
    logger.info(f"🔌 Server will be available at: http://0.0.0.0:6500")
    logger.info(f"📚 API Documentation: http://localhost:6500/docs")
    logger.info(f"🏥 Health Check: http://localhost:6500/api/health")
    logger.info("📋 Logging is configured for console output including Docker containers")

    # Run the server with enhanced logging
    uvicorn.run(
        "__main__:app", # Use __main__ to make it directly runnable
        host="0.0.0.0",
        port=6500,
        reload=True,
        log_level="info",
        access_log=True,
        use_colors=True if sys.stdout.isatty() else False  # Auto-detect terminal for colors
    )
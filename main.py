#!/usr/bin/env python3
"""
ORION Medical AI System - Main Application Entry Point

A comprehensive medical imaging analysis platform that provides:
- DICOM image processing and visualization
- AI-powered medical image segmentation using MedSAM
- Anatomical structure analysis and ROI detection
- 3D mesh generation from medical volumes
- Real-time medical image analysis APIs

This modular system is designed for medical research and analysis workflows.
"""

import os
import json
import logging
import asyncio
import gc
import math
import time
import uuid
from collections import defaultdict
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union, Literal, Optional, Tuple

import psutil
import uvicorn
import numpy as np
import cv2
import requests
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Path, Body, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from skimage.measure import regionprops
from skimage import measure

# Import our modularized components
from modules.models import (
    Point, RectangularROI, CircularROI, SeriesInfo, LoadSeriesResponse,
    MeshGenerationStatus, RectangleROIRequest, CircleROIRequest
)
from modules.ai_core import LocalMedSAM, LocalROIAnalyzer
from modules.cache_storage import DiskBasedGlobalContextCache, VectorDBManager, BackendCache
from modules.utils import (
    analyze_roi_anatomical_overlap, find_overlapping_structures,
    get_gcloud_access_token, check_and_upload_slice_to_gcs,
    scan_for_series, resolve_safe_path, load_dicom_volume,
    preprocess_volume_for_3d
)

# Optional 3D mesh processing
try:
    import trimesh
except ImportError as e:
    print(f"Failed to import trimesh: {e}")
    trimesh = None

# Load environment variables
load_dotenv()

# Configure logging
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

# Configuration constants
CANVAS_WIDTH = 512
CANVAS_HEIGHT = 512
CANVAS_SIZE = (CANVAS_WIDTH, CANVAS_HEIGHT)
IMAGE_TARGET_SIZE = (512, 512)
COORDINATE_SCALE = 512

# DICOM data directory configuration - portable across environments
DICOM_DATA_ROOT = os.getenv("DICOM_DATA_ROOT", "/")  # for testing in the ORION directory
logger.info(f"DICOM data root configured to: {DICOM_DATA_ROOT}")


class CustomJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        def custom_serializer(obj):
            if isinstance(obj, (np.ndarray,)): return obj.tolist()
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            try: return json.JSONEncoder().default(obj)
            except: return str(obj)
        return json.dumps(content, default=custom_serializer, indent=2).encode()


# FastAPI app configuration
app = FastAPI(
    title="ORION Medical AI Backend",
    description="Advanced medical imaging analysis platform with AI-powered segmentation and anatomical analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=CustomJSONResponse,
    openapi_tags=[
        {"name": "health", "description": "Health check and system monitoring"},
        {"name": "dicom", "description": "DICOM file operations and series management"},
        {"name": "analysis", "description": "Medical image analysis and AI inference"},
        {"name": "3d", "description": "3D mesh generation and visualization"},
        {"name": "cache", "description": "Cache management operations"},
        {"name": "system", "description": "System administration and utilities"}
    ]
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables and managers
current_volume = None
current_series_uid = None
current_dicom_metadata = {}
anatomical_context_cache = {}
system_metrics = {
    "requests_processed": 0,
    "total_processing_time": 0,
    "last_request_time": None,
    "active_sessions": 0,
    "memory_peak_usage": 0
}

# Initialize managers
global_context_cache = DiskBasedGlobalContextCache()
vector_db = VectorDBManager()
backend_cache = BackendCache()
mesh_status_manager = MeshGenerationStatus()

# AI model instances (initialized on startup)
medsam_model = None
roi_analyzer = None


@app.on_event("startup")
async def startup_event():
    """Initialize AI models and system components on startup."""
    global medsam_model, roi_analyzer

    logger.info("🚀 Starting ORION Medical AI Backend...")

    # Initialize AI models
    logger.info("🤖 Initializing AI models...")
    medsam_model = LocalMedSAM()
    roi_analyzer = LocalROIAnalyzer()

    # Load models in background
    try:
        medsam_model.load_model()
        roi_analyzer.load_model()
        logger.info("✅ AI models loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Some AI models failed to load: {e}")

    # Initialize vector database
    try:
        vector_db.load_database()
        logger.info("✅ Vector database initialized")
    except Exception as e:
        logger.warning(f"⚠️ Vector database initialization failed: {e}")

    logger.info("🎉 ORION Medical AI Backend startup complete!")


# API Routes
@app.get("/", tags=["health"])
async def root():
    return {"message": "ORION Medical AI Backend", "status": "running", "version": "2.0.0"}


@app.get("/api/health", tags=["health"])
async def health_check():
    """Comprehensive health check endpoint."""
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "memory_usage": f"{memory_info.percent}%",
            "available_memory": f"{memory_info.available / (1024**3):.2f} GB",
            "disk_usage": f"{disk_info.percent}%",
            "available_disk": f"{disk_info.free / (1024**3):.2f} GB"
        },
        "ai_models": {
            "medsam_ready": medsam_model.is_ready if medsam_model else False,
            "roi_analyzer_ready": roi_analyzer.is_ready if roi_analyzer else False
        },
        "metrics": system_metrics,
        "cache_stats": {
            "global_context_entries": len(global_context_cache.memory_cache),
            "backend_cache_entries": len(backend_cache.global_context_cache)
        }
    }


@app.get("/api/metrics", tags=["health"])
async def get_system_metrics():
    """Get detailed system performance metrics."""
    process = psutil.Process()

    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_info": {
            "rss": process.memory_info().rss / (1024**2),  # MB
            "vms": process.memory_info().vms / (1024**2),  # MB
            "percent": process.memory_percent()
        },
        "gpu_info": {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
        },
        "request_metrics": system_metrics,
        "cache_status": {
            "disk_cache_size": len(global_context_cache.memory_cache),
            "memory_cache_size": len(backend_cache.global_context_cache)
        }
    }


if __name__ == "__main__":
    # Log startup information
    logger.info("🚀 Starting Standalone Medical AI Backend")
    logger.info(f"🔌 Server will be available at: http://0.0.0.0:6500")
    logger.info(f"📚 API Documentation: http://localhost:6500/docs")
    logger.info(f"🏥 Health Check: http://localhost:6500/api/health")
    logger.info("📋 Logging is configured for console output including Docker containers")

    uvicorn.run(
        "testing:app",
        host="0.0.0.0",
        port=6500,
        reload=False,
        log_level="info",
        access_log=True
    )
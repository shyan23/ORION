"""
Utility functions for the ORION Medical AI System
"""

import os
import math
import logging
import tempfile
import hashlib
from io import BytesIO
from typing import List, Dict, Any, Union, Tuple, Optional
import numpy as np
import cv2
import pydicom
import requests
from pathlib import Path
import google.auth
import google.auth.transport.requests
from google.cloud import storage
from .models import Point, RectangularROI, CircularROI

logger = logging.getLogger(__name__)

COORDINATE_SCALE = 512  # Scale for converting between canvas and volume coordinates


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


def calculate_overlap(roi_region: Union[RectangularROI, CircularROI], structure_bbox: RectangularROI) -> Dict:
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
    """Calculate the distance from a point to the center of a bounding box."""
    center_x = (bbox.get("x_min", 0) + bbox.get("x_max", 1)) / 2
    center_y = (bbox.get("y_min", 0) + bbox.get("y_max", 1)) / 2
    return math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)


def get_gcloud_access_token():
    """Get Google Cloud access token for authentication."""
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
    """Upload a slice to Google Cloud Storage if not already present."""
    try:
        bucket_name = os.getenv("GCS_BUCKET_NAME", "orion-medical-data")
        object_name = f"slices/{series_uid}/{slice_id}.png"

        # Get access token
        access_token = get_gcloud_access_token()
        if not access_token:
            return None

        # Check if file exists
        check_url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_name.replace('/', '%2F')}"
        headers = {"Authorization": f"Bearer {access_token}"}

        check_response = requests.get(check_url, headers=headers, timeout=10)

        if check_response.status_code == 200:
            logger.info(f"Slice {slice_id} already exists in GCS")
            return f"gs://{bucket_name}/{object_name}"

        # Upload if doesn't exist
        normalized_slice = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255).astype(np.uint8)

        # Convert to PNG bytes
        import cv2
        success, img_encoded = cv2.imencode('.png', normalized_slice)
        if not success:
            logger.error("Failed to encode image as PNG")
            return None

        png_bytes = img_encoded.tobytes()

        # Upload to GCS
        upload_url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={object_name}"
        upload_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "image/png"
        }

        upload_response = requests.post(upload_url, headers=upload_headers, data=png_bytes, timeout=30)

        if upload_response.status_code == 200:
            logger.info(f"Successfully uploaded slice {slice_id} to GCS")
            return f"gs://{bucket_name}/{object_name}"
        else:
            logger.error(f"Failed to upload to GCS: {upload_response.status_code} - {upload_response.text}")
            return None

    except Exception as e:
        logger.error(f"Error uploading to GCS: {e}")
        return None


def scan_for_series(dir_path):
    """Scan directory for DICOM series."""
    series_dict = {}
    dicom_files = []

    try:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return {"error": f"Directory not found: {dir_path}"}

        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                try:
                    ds = pydicom.dcmread(str(file_path), force=True)
                    if hasattr(ds, 'SeriesInstanceUID'):
                        dicom_files.append((str(file_path), ds))
                except Exception:
                    continue

        for file_path, ds in dicom_files:
            series_uid = ds.SeriesInstanceUID

            if series_uid not in series_dict:
                series_dict[series_uid] = {
                    "series_uid": series_uid,
                    "modality": getattr(ds, 'Modality', 'Unknown'),
                    "description": getattr(ds, 'SeriesDescription', 'No Description'),
                    "study_date": getattr(ds, 'StudyDate', 'Unknown'),
                    "patient_id": getattr(ds, 'PatientID', 'Unknown'),
                    "study_description": getattr(ds, 'StudyDescription', 'No Description'),
                    "files": [],
                    "slice_count": 0
                }

            series_dict[series_uid]["files"].append(file_path)
            series_dict[series_uid]["slice_count"] += 1

        # Add first slice path for each series
        for series_uid, series_info in series_dict.items():
            if series_info["files"]:
                series_info["first_slice_path"] = series_info["files"][0]

        return list(series_dict.values())

    except Exception as e:
        logger.error(f"Error scanning directory: {e}")
        return {"error": str(e)}


def resolve_safe_path(user_path):
    """Resolve and validate user-provided path for safety."""
    try:
        # Handle different path formats
        if user_path.startswith('file://'):
            user_path = user_path[7:]

        # Convert to Path object and resolve
        path = Path(user_path).resolve()

        # Basic safety checks
        path_str = str(path)

        # Ensure it's an absolute path
        if not path.is_absolute():
            return None, "Path must be absolute"

        # Check if path exists
        if not path.exists():
            return None, f"Path does not exist: {path_str}"

        # Check if it's a directory
        if not path.is_dir():
            return None, f"Path is not a directory: {path_str}"

        logger.info(f"Resolved safe path: {path_str}")
        return str(path), None

    except Exception as e:
        logger.error(f"Path resolution error: {e}")
        return None, f"Invalid path: {str(e)}"


def load_dicom_volume(files):
    """Load DICOM files and return volume array."""
    dicom_slices = []

    for file_path in files:
        try:
            ds = pydicom.dcmread(file_path)
            dicom_slices.append(ds)
        except Exception as e:
            logger.warning(f"Failed to read DICOM file {file_path}: {e}")
            continue

    if not dicom_slices:
        return None

    # Sort slices by SliceLocation or InstanceNumber
    dicom_slices.sort(key=lambda x: float(getattr(x, 'SliceLocation', getattr(x, 'InstanceNumber', 0))))

    # Extract pixel arrays
    volume = np.stack([ds.pixel_array for ds in dicom_slices], axis=0)

    return volume


def preprocess_volume_for_3d(volume: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Preprocess volume for 3D visualization with windowing."""
    # Apply windowing
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2

    # Clip and normalize
    windowed = np.clip(volume, window_min, window_max)
    normalized = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)

    return normalized
"""
Data models and structures for the ORION Medical AI System
"""

import math
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union, Literal, Optional, Tuple, NamedTuple
from pydantic import BaseModel, validator, Field
import numpy as np


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

    def get_area(self) -> float:
        return math.pi * self.radius**2


class SeriesInfo(BaseModel):
    series_uid: str
    modality: str
    description: str
    slice_count: int
    study_date: Optional[str] = None
    patient_id: Optional[str] = None
    study_description: Optional[str] = None
    first_slice_path: str


class LoadSeriesResponse(BaseModel):
    series_uid: str
    slice_count: int
    canvas_size: Tuple[int, int] = (512, 512)


class MeshGenerationStatus:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()

    def start_job(self, job_id: str, series_uid: str):
        with self.lock:
            self.jobs[job_id] = {
                'status': 'processing',
                'series_uid': series_uid,
                'start_time': datetime.now(),
                'progress': 0,
                'message': 'Starting mesh generation...'
            }

    def update_job(self, job_id: str, progress: int, message: str):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['progress'] = progress
                self.jobs[job_id]['message'] = message

    def complete_job(self, job_id: str, mesh_file: str = None, error: str = None):
        with self.lock:
            if job_id in self.jobs:
                if error:
                    self.jobs[job_id]['status'] = 'failed'
                    self.jobs[job_id]['error'] = error
                else:
                    self.jobs[job_id]['status'] = 'completed'
                    self.jobs[job_id]['mesh_file'] = mesh_file
                self.jobs[job_id]['end_time'] = datetime.now()

    def get_job_status(self, job_id: str):
        with self.lock:
            return self.jobs.get(job_id)

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        with self.lock:
            jobs_to_remove = []
            for job_id, job in self.jobs.items():
                if 'end_time' in job and job['end_time'] < cutoff_time:
                    jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self.jobs[job_id]


class RectangleROIRequest(BaseModel):
    series_uid: str
    slice_index: int
    top_left_x: float = Field(..., ge=0, le=512)
    top_left_y: float = Field(..., ge=0, le=512)
    bottom_right_x: float = Field(..., ge=0, le=512)
    bottom_right_y: float = Field(..., ge=0, le=512)


class CircleROIRequest(BaseModel):
    series_uid: str
    slice_index: int
    center_x: float = Field(..., ge=0, le=512)
    center_y: float = Field(..., ge=0, le=512)
    radius: float = Field(..., ge=1, le=256)
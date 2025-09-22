# Agentic AI Surgical Assistant - Python Module Architecture

## Project Structure Overview

```
orion-ai-surgical-assistant/
├── backend/                          # Python FastAPI Backend
│   ├── modules/                      # Core Business Logic Modules
│   ├── api/                         # REST API Layer
│   ├── services/                    # External Service Integrations
│   ├── database/                    # Database Models & Migrations
│   └── tests/                       # Test Suite
├── frontend/                        # React/Three.js Frontend
├── ml-services/                     # Containerized ML Models
├── docker/                          # Docker Configurations
└── deployment/                      # Kubernetes/Infrastructure
```

---

# Backend Module Architecture (Python)

## Core Module Structure

```
backend/
├── main.py                          # FastAPI Application Entry Point
├── config/
│   ├── __init__.py
│   ├── settings.py                  # Environment Configuration
│   └── database.py                  # Database Connection Setup
├── modules/                         # Business Logic Modules
│   ├── __init__.py
│   ├── foundation/                  # Layer 1: Foundation
│   ├── ml_processing/               # Layer 2: ML Services
│   ├── tools/                       # Layer 3: Agent Tools
│   ├── reasoning/                   # Layer 4: Agent Brain
│   └── collaboration/               # Real-time Features
├── api/                            # FastAPI Route Handlers
├── services/                       # External Service Clients
├── database/                       # SQLAlchemy Models
├── utils/                          # Shared Utilities
└── tests/                          # Unit & Integration Tests
```

---

## Module 1: Foundation Layer (`modules/foundation/`)

**Purpose**: Core data management and 3D visualization backend support

```python
modules/foundation/
├── __init__.py
├── dicom/
│   ├── __init__.py
│   ├── parser.py              # DICOM file parsing
│   ├── processor.py           # DICOM data processing
│   └── storage.py             # DICOM file storage management
├── spatial/
│   ├── __init__.py
│   ├── coordinates.py         # 3D coordinate systems
│   ├── mesh_utils.py          # 3D mesh operations
│   └── transforms.py          # Spatial transformations
├── session/
│   ├── __init__.py
│   ├── manager.py             # Multi-user session management
│   └── state.py               # Session state persistence
└── visualization/
    ├── __init__.py
    ├── renderer_api.py        # 3D renderer backend support
    └── export.py              # 3D data export utilities
```

### Key Classes:

```python
# modules/foundation/dicom/parser.py
class DICOMParser:
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse DICOM file and extract metadata"""
    
    def extract_3d_data(self, dicom_data: Dict) -> np.ndarray:
        """Convert DICOM to 3D numpy array"""

# modules/foundation/spatial/coordinates.py  
class SpatialCoordinator:
    def world_to_voxel(self, world_coords: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert world coordinates to voxel indices"""
    
    def calculate_distance_3d(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Calculate 3D Euclidean distance"""
```

---

## Module 2: ML Processing (`modules/ml_processing/`)

**Purpose**: Medical image analysis and computer vision

```python
modules/ml_processing/
├── __init__.py
├── segmentation/
│   ├── __init__.py
│   ├── models.py              # ML model wrappers
│   ├── inference.py           # Segmentation inference
│   └── postprocess.py         # Segmentation post-processing
├── analysis/
│   ├── __init__.py
│   ├── structure_detector.py  # Anatomical structure detection
│   ├── margin_calculator.py   # Distance calculations
│   └── risk_assessor.py       # Surgical risk analysis
├── preprocessing/
│   ├── __init__.py
│   ├── normalization.py       # Image normalization
│   ├── registration.py        # Image registration
│   └── augmentation.py        # Data augmentation
└── model_management/
    ├── __init__.py
    ├── loader.py              # Model loading/caching
    ├── versioning.py          # Model version control
    └── deployment.py          # Model deployment utilities
```

### Key Classes:

```python
# modules/ml_processing/segmentation/inference.py
class SegmentationInference:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
    
    def segment_structures(self, image_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment anatomical structures from 3D image"""
    
    def identify_at_point(self, image_data: np.ndarray, coordinates: Tuple[int, int, int]) -> str:
        """Identify structure at specific 3D coordinates"""

# modules/ml_processing/analysis/margin_calculator.py
class MarginCalculator:
    def calculate_minimum_distance(self, structure_a: np.ndarray, structure_b: np.ndarray) -> float:
        """Calculate minimum distance between two 3D structures"""
    
    def assess_surgical_margin(self, tumor: np.ndarray, critical_structures: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Assess surgical margins for multiple structures"""
```

---

## Module 3: Agent Tools (`modules/tools/`)

**Purpose**: Atomic functions the agent can execute

```python
modules/tools/
├── __init__.py
├── base/
│   ├── __init__.py
│   ├── tool_interface.py      # Abstract base class for tools
│   ├── tool_registry.py       # Tool registration system
│   └── validation.py          # Input/output validation
├── spatial_tools/
│   ├── __init__.py
│   ├── structure_identifier.py
│   ├── margin_calculator.py
│   └── proximity_analyzer.py
├── rag_tools/
│   ├── __init__.py
│   ├── protocol_lookup.py
│   ├── patient_search.py
│   └── knowledge_retriever.py
├── visualization_tools/
│   ├── __init__.py
│   ├── structure_highlighter.py
│   └── annotation_manager.py
└── integration/
    ├── __init__.py
    ├── tool_executor.py       # Tool execution engine
    └── result_processor.py    # Tool result processing
```

### Key Implementation:

```python
# modules/tools/base/tool_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from pydantic import BaseModel

class ToolInput(BaseModel):
    """Base class for tool inputs with validation"""
    pass

class ToolOutput(BaseModel):
    """Base class for tool outputs with validation"""
    success: bool
    data: Any
    message: str = ""

class BaseTool(ABC):
    """Abstract base class for all agent tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for agent reference"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for agent understanding"""
        pass
    
    @abstractmethod
    def execute(self, input_data: ToolInput) -> ToolOutput:
        """Execute the tool with given input"""
        pass

# modules/tools/spatial_tools/structure_identifier.py
class StructureIdentifierTool(BaseTool):
    name = "identify_structure_at_point"
    description = "Identifies anatomical structure at 3D coordinates from user click"
    
    def __init__(self, ml_service: SegmentationInference):
        self.ml_service = ml_service
    
    def execute(self, input_data: Dict[str, Any]) -> ToolOutput:
        coordinates = input_data["coordinates"]
        patient_id = input_data["patient_id"]
        
        # Get patient's image data
        image_data = self._load_patient_image(patient_id)
        
        # Identify structure
        structure_name = self.ml_service.identify_at_point(image_data, coordinates)
        
        return ToolOutput(
            success=True,
            data={"structure_name": structure_name, "coordinates": coordinates},
            message=f"Structure identified as '{structure_name}'"
        )
```

---

## Module 4: Reasoning Engine (`modules/reasoning/`)

**Purpose**: The autonomous agent brain using ReAct framework

```python
modules/reasoning/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── agent.py               # Main ReAct Agent implementation
│   ├── planner.py             # Goal decomposition logic
│   └── executor.py            # Action execution controller
├── prompts/
│   ├── __init__.py
│   ├── base_prompts.py        # Core system prompts
│   ├── medical_prompts.py     # Medical domain-specific prompts
│   └── few_shot_examples.py   # Example cases for learning
├── memory/
│   ├── __init__.py
│   ├── conversation_memory.py # Conversation context management
│   ├── working_memory.py      # Short-term reasoning memory
│   └── knowledge_cache.py     # Cached knowledge retrieval
└── synthesis/
    ├── __init__.py
    ├── report_generator.py    # Final report synthesis
    └── insight_extractor.py   # Key insight extraction
```

### Key Implementation:

```python
# modules/reasoning/core/agent.py
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool as LangChainTool

class ReActSurgicalAgent:
    """Main ReAct agent for surgical assessment"""
    
    def __init__(self, 
                 llm: Any,  # Language model instance
                 tools: List[BaseTool],
                 memory_manager: 'ConversationMemory'):
        self.llm = llm
        self.tools = self._convert_tools_to_langchain(tools)
        self.memory = memory_manager
        self.agent_executor = self._build_agent()
    
    def assess_resectability(self, patient_id: str, user_query: str) -> Dict[str, Any]:
        """Main entry point for surgical assessment"""
        
        # Add patient context to memory
        self.memory.add_patient_context(patient_id)
        
        # Execute agent reasoning
        result = self.agent_executor.run(
            input=user_query,
            patient_id=patient_id
        )
        
        return {
            "assessment": result,
            "reasoning_trace": self.memory.get_reasoning_trace(),
            "tools_used": self.memory.get_tools_used()
        }

# modules/reasoning/core/planner.py
class GoalDecomposer:
    """Decomposes high-level goals into executable steps"""
    
    def decompose_resectability_assessment(self, patient_data: Dict) -> List[str]:
        """Create step-by-step plan for resectability assessment"""
        
        base_steps = [
            "1. Identify the primary tumor location and characteristics",
            "2. Identify all adjacent critical structures (vessels, ducts, organs)",
            "3. Measure surgical margins between tumor and critical structures", 
            "4. Retrieve patient-specific risk factors from medical records",
            "5. Look up standard surgical protocols for this tumor type",
            "6. Synthesize all findings into comprehensive risk assessment"
        ]
        
        # Customize based on patient data
        return self._customize_plan(base_steps, patient_data)
```

---

## Module 5: Collaboration (`modules/collaboration/`)

**Purpose**: Real-time multi-user features and WebSocket management

```python
modules/collaboration/
├── __init__.py
├── websocket/
│   ├── __init__.py
│   ├── manager.py             # WebSocket connection management
│   ├── handlers.py            # WebSocket event handlers
│   └── broadcasting.py        # Multi-user broadcasting
├── session/
│   ├── __init__.py
│   ├── room_manager.py        # Surgery session rooms
│   ├── user_manager.py        # User presence tracking
│   └── state_sync.py          # Synchronized state management
└── streaming/
    ├── __init__.py
    ├── agent_stream.py        # Real-time agent reasoning stream
    └── data_stream.py         # 3D visualization data streaming
```

---

## API Layer (`api/`)

**Purpose**: FastAPI route handlers that expose module functionality

```python
api/
├── __init__.py
├── v1/
│   ├── __init__.py
│   ├── patients.py            # Patient management endpoints
│   ├── imaging.py             # Medical imaging endpoints  
│   ├── assessment.py          # Surgical assessment endpoints
│   ├── collaboration.py       # Real-time collaboration endpoints
│   └── tools.py               # Direct tool execution endpoints
├── websocket/
│   ├── __init__.py
│   └── routes.py              # WebSocket endpoints
└── middleware/
    ├── __init__.py
    ├── auth.py                # Authentication middleware
    ├── cors.py                # CORS configuration
    └── rate_limiting.py       # API rate limiting
```

### Key API Endpoints:

```python
# api/v1/assessment.py
from fastapi import APIRouter, Depends, BackgroundTasks
from modules.reasoning.core.agent import ReActSurgicalAgent

router = APIRouter(prefix="/assessment", tags=["surgical-assessment"])

@router.post("/resectability")
async def assess_resectability(
    request: ResectabilityRequest,
    background_tasks: BackgroundTasks,
    agent: ReActSurgicalAgent = Depends(get_agent)
):
    """Trigger autonomous resectability assessment"""
    
    # Start agent reasoning in background
    result = await agent.assess_resectability(
        patient_id=request.patient_id,
        user_query=request.query
    )
    
    return {
        "assessment_id": result["assessment_id"],
        "status": "processing",
        "reasoning_trace": result["reasoning_trace"]
    }

@router.websocket("/stream/{assessment_id}")
async def stream_agent_reasoning(websocket: WebSocket, assessment_id: str):
    """Stream real-time agent reasoning process"""
    await websocket.accept()
    
    # Stream agent's internal monologue
    async for reasoning_step in agent_stream.get_reasoning_stream(assessment_id):
        await websocket.send_json(reasoning_step)
```

---

## Database Models (`database/`)

**Purpose**: SQLAlchemy models for data persistence

```python
database/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── patient.py             # Patient information
│   ├── imaging.py             # Medical imaging data
│   ├── assessment.py          # Assessment results
│   ├── session.py             # Collaboration sessions
│   └── audit.py               # Audit trails
├── migrations/                # Alembic database migrations
└── repositories/
    ├── __init__.py
    ├── patient_repository.py
    ├── assessment_repository.py
    └── base_repository.py
```

---

## Services Layer (`services/`)

**Purpose**: External service integrations

```python
services/
├── __init__.py
├── llm/
│   ├── __init__.py
│   ├── openai_client.py       # OpenAI integration
│   ├── anthropic_client.py    # Claude integration
│   └── local_llm_client.py    # Local LLM support
├── vector_db/
│   ├── __init__.py
│   ├── pinecone_client.py     # Vector database client
│   └── embedding_service.py   # Text embedding service
├── storage/
│   ├── __init__.py
│   ├── s3_client.py           # AWS S3 integration
│   └── local_storage.py       # Local file storage
└── ml_models/
    ├── __init__.py
    ├── model_client.py        # ML model API client
    └── inference_service.py   # Model inference service
```

---

## Development Workflow

### Phase 1 Setup (Week 1-2):
```bash
# Initialize project
poetry init
poetry add fastapi uvicorn sqlalchemy alembic pydantic
poetry add numpy scipy scikit-image
poetry add langchain openai anthropic
poetry add pytest pytest-asyncio

# Create basic structure
python scripts/create_modules.py
alembic init alembic
```

### Phase 2 Core Implementation (Week 3-6):
```python
# Start with foundation module
python -m modules.foundation.dicom.parser
python -m modules.tools.spatial_tools.structure_identifier  
python -m modules.reasoning.core.agent

# Test each module independently
pytest tests/modules/foundation/
pytest tests/modules/tools/
pytest tests/modules/reasoning/
```

### Phase 3 Integration (Week 7-8):
```python
# Wire up API layer
python -m api.v1.assessment
python -m api.websocket.routes

# End-to-end testing
pytest tests/integration/
```

This modular architecture provides clear separation of concerns, making it easy to develop and test each component independently while maintaining the sophisticated agent capabilities described in your original document.
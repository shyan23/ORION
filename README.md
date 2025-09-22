# ORION Medical AI System

A comprehensive medical imaging analysis platform that provides AI-powered medical image segmentation, anatomical structure analysis, and 3D visualization capabilities.

## 🏥 Overview

ORION is a modular medical AI system designed for medical research and clinical analysis workflows. It combines state-of-the-art AI models with robust medical imaging processing capabilities to provide:

- **DICOM Processing**: Complete DICOM file handling and series management
- **AI Segmentation**: MedSAM-powered medical image segmentation
- **Anatomical Analysis**: Real-time anatomical structure detection and ROI analysis
- **3D Visualization**: Volume rendering and mesh generation from medical images
- **Cache Management**: Intelligent caching for improved performance
- **RESTful API**: Complete FastAPI-based backend for integration

## 🏗️ Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORION Medical AI System                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   Frontend UI   │    │   API Gateway   │    │  Admin Panel │ │
│  │   (External)    │    │   (FastAPI)     │    │  (Optional)  │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                      │      │
│           └───────────────────────┼──────────────────────┘      │
│                                   │                             │
├───────────────────────────────────┼─────────────────────────────┤
│                                   ▼                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 Main Application (testing.py)               │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │ │
│  │  │   Models    │  │  AI Core    │  │ Cache/Store │         │ │
│  │  │             │  │             │  │             │         │ │
│  │  │ • Data      │  │ • MedSAM    │  │ • Disk Cache│         │ │
│  │  │   Models    │  │ • ROI       │  │ • Vector DB │         │ │
│  │  │ • Pydantic  │  │   Analyzer  │  │ • Memory    │         │ │
│  │  │   Schemas   │  │ • AI Models │  │   Cache     │         │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │ │
│  │                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │                 Utilities Module                        │ │ │
│  │  │                                                         │ │ │
│  │  │ • DICOM Processing    • 3D Mesh Generation             │ │ │
│  │  │ • Image Analysis      • Google Cloud Integration       │ │ │
│  │  │ • ROI Calculations    • File System Operations         │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                     External Dependencies                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │  PyTorch/CUDA   │  │  Google Cloud   │  │  File System    │   │
│  │  (AI Models)    │  │   Storage       │  │   (DICOM)       │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   DICOM     │────▶│   Volume    │────▶│  AI Model   │
│   Input     │     │ Processing  │     │ Inference   │
└─────────────┘     └─────────────┘     └─────────────┘
                             │                   │
                             ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Results    │◀────│    Cache    │◀────│ Anatomical  │
│   Output    │     │ Management  │     │  Analysis   │
└─────────────┘     └─────────────┘     └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for accelerated AI inference)
- At least 8GB RAM
- 10GB free disk space

### Installation

1. **Clone the repository**
   ```bash
   cd /path/to/your/workspace
   git clone <repository-url>
   cd ORION
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python testing.py
   ```

The server will start on `http://localhost:6500`

## 📁 Project Structure

```
ORION/
├── testing.py                 # Main application entry point
├── testing_original_backup.py # Original monolithic version
├── requirements.txt           # Python dependencies
├── .env                      # Environment configuration
├── .gitignore               # Git ignore rules
├── README.md                # This file
│
├── modules/                 # Modularized components
│   ├── __init__.py
│   ├── models.py           # Data models and Pydantic schemas
│   ├── ai_core.py          # AI models (MedSAM, ROI Analyzer)
│   ├── cache_storage.py    # Cache and storage management
│   └── utils.py            # Utility functions and helpers
│
├── ai_models/              # AI model weights and configs
│   └── Swin_medsam/
│       └── model.pth
│
├── Swin_LiteMedSAM/        # MedSAM model architecture
│   ├── models/
│   │   ├── mask_decoder.py
│   │   ├── prompt_encoder.py
│   │   ├── swin.py
│   │   └── transformer.py
│   └── ...
│
├── cache/                  # Persistent cache storage
│   └── global_context/
│
├── vector_db/             # Vector database for RAG
├── static/                # Static files
├── uploads/               # File uploads
└── frontend/              # Frontend application (if applicable)
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# DICOM Data Configuration
DICOM_DATA_ROOT=/path/to/dicom/data

# Google Cloud Storage (Optional)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
GCS_BUCKET_NAME=your-bucket-name

# Cache Settings
CACHE_TTL_HOURS=168
MAX_CACHE_SIZE_MB=500

# API Configuration
API_HOST=0.0.0.0
API_PORT=6500
LOG_LEVEL=INFO
```

## 🏃‍♂️ Running the System

### Development Mode

```bash
# Run with auto-reload
python testing.py

# Or with uvicorn directly
uvicorn testing:app --host 0.0.0.0 --port 6500 --reload
```

### Production Mode

```bash
# Run with optimized settings
uvicorn testing:app --host 0.0.0.0 --port 6500 --workers 4
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 6500

CMD ["python", "testing.py"]
```

```bash
# Build and run
docker build -t orion-medical-ai .
docker run -p 6500:6500 orion-medical-ai
```

## 📖 API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:6500/docs
- **ReDoc**: http://localhost:6500/redoc

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System health check |
| `/api/scan-directory` | GET | Scan for DICOM series |
| `/api/load-series` | POST | Load DICOM series |
| `/api/analyze-roi` | POST | Analyze region of interest |
| `/api/generate-global-context` | POST | Generate anatomical context |
| `/api/3d/generate-mesh/{series_uid}` | POST | Generate 3D mesh |

### Example API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:6500/api/health")
print(response.json())

# Scan for DICOM series
response = requests.get("http://localhost:6500/api/scan-directory?dir_path=/path/to/dicom")
series = response.json()

# Load a series
payload = {"series_uid": "1.2.3.4.5", "files": ["file1.dcm", "file2.dcm"]}
response = requests.post("http://localhost:6500/api/load-series", json=payload)
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=modules tests/

# Run specific test file
pytest tests/test_models.py
```

### Integration Tests

```bash
# Test API endpoints
pytest tests/test_api.py

# Test AI models
pytest tests/test_ai_core.py
```

## 🔍 Monitoring and Logging

### Health Monitoring

- **Health Check**: `/api/health`
- **Metrics**: `/api/metrics`
- **System Stats**: Real-time memory, CPU, and GPU monitoring

### Logging

Logs are configured to output to stdout/stderr for Docker compatibility:

```python
# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO  # Set in .env
```

### Performance Monitoring

The system tracks:
- Request processing times
- Memory usage patterns
- Cache hit/miss ratios
- AI model inference times

## 🧠 AI Models

### MedSAM (Medical Segment Anything Model)

- **Purpose**: Medical image segmentation
- **Input**: Medical images + point/box prompts
- **Output**: Segmentation masks
- **Location**: `modules/ai_core.py:LocalMedSAM`
- **Model Download**: Download the Swin_LiteMedSAM model from https://github.com/RuochenGao/Swin_LiteMedSAM

### ROI Analyzer (BiomedCLIP)

- **Purpose**: Anatomical content analysis
- **Input**: Region of interest images
- **Output**: Anatomical classification
- **Location**: `modules/ai_core.py:LocalROIAnalyzer`

## 💾 Data Management

### Cache Strategy

1. **Memory Cache**: Fast access for active sessions
2. **Disk Cache**: Persistent storage for global context
3. **Vector Database**: Semantic search for anatomical data

### Storage Locations

- **Local Cache**: `cache/global_context/`
- **Vector DB**: `vector_db/`
- **AI Models**: `ai_models/`
- **DICOM Data**: Configurable via `DICOM_DATA_ROOT`

## 🔒 Security Considerations

- DICOM data is processed locally by default
- No sensitive data in logs
- Service account keys should be properly secured
- API rate limiting recommended for production

## 🚨 Troubleshooting

### Common Issues

1. **AI Models Not Loading**
   ```bash
   # Check model files exist
   ls -la ai_models/Swin_medsam/

   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **DICOM Files Not Found**
   ```bash
   # Check DICOM_DATA_ROOT setting
   echo $DICOM_DATA_ROOT

   # Verify directory permissions
   ls -la /path/to/dicom/data
   ```

3. **Memory Issues**
   ```bash
   # Monitor memory usage
   watch -n 1 'free -h'

   # Clear caches
   curl -X DELETE http://localhost:6500/api/cache/global-context/clear
   ```

### Performance Optimization

1. **Enable GPU**: Install CUDA-compatible PyTorch
2. **Increase Cache Size**: Adjust `MAX_CACHE_SIZE_MB`
3. **Use SSD Storage**: For better I/O performance
4. **Scale Horizontally**: Deploy multiple instances

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the logs for error details

---

**ORION Medical AI System** - Advancing medical imaging through AI innovation 🏥✨
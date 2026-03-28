# Docker Quick Start Guide

Quick reference for Docker deployment of Immo Eliza ML.

## Prerequisites

1. **Install Docker:**
   - [Docker Desktop](https://www.docker.com/products/docker-desktop) (Mac/Windows)
   - Or Docker Engine (Linux)

2. **Train models first:**
   ```bash
   python -m immo_eliza_ml.main
   ```
   This creates the `models/` folder.

## Quick Commands

### Run Both Services (Recommended)

```bash
# Start both Streamlit and FastAPI
docker-compose up --build

# Or in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Access:**
- 🌐 Streamlit: http://localhost:8501
- 🔌 API: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs

### Streamlit Only

```bash
# Build
docker build -f Dockerfile.streamlit -t immo-streamlit .

# Run
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  immo-streamlit
```

### FastAPI Only

```bash
# Build
docker build -f Dockerfile.api -t immo-api .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  immo-api
```

## Test the Containers

### Test Streamlit
```bash
curl http://localhost:8501
```

### Test API
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "postal_code": 9000,
    "living_area": 120,
    "number_of_rooms": 3,
    "number_of_facades": 2,
    "equipped_kitchen": 1,
    "garden": 1,
    "garden_surface": 150
  }'
```

## Troubleshooting

### Models not found
- Ensure `models/` folder exists with trained models
- Check volume mounting: `-v $(pwd)/models:/app/models:ro`

### Port already in use
```bash
# Change ports in docker-compose.yml or use different ports:
docker run -p 8502:8501 immo-streamlit
```

### View container logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs streamlit
docker-compose logs api

# Follow logs
docker-compose logs -f
```

### Rebuild after code changes
```bash
docker-compose up --build --force-recreate
```

## Production Deployment

See `DEPLOYMENT.md` for cloud deployment options (Google Cloud Run, AWS, Azure, etc.)


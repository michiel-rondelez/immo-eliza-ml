# Deployment Guide

This guide covers deploying the Immo Eliza ML application as a web service.

## Prerequisites

1. **Train your models first:**
   ```bash
   python -m immo_eliza_ml.main
   ```
   This creates the `models/` folder with trained models.

2. **Install dependencies:**
   ```bash
   poetry install
   # or
   pip install -r requirements.txt
   ```

---

## Option 1: Streamlit App (Easiest - Recommended)

### Local Testing

```bash
streamlit run app.py
```

Visit `http://localhost:8501`

### Deploy to Streamlit Cloud (Free)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Streamlit app"
   git push origin main
   ```

2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file path: `app.py`
   - Click "Deploy"

3. **Your app will be live at:** `https://your-app-name.streamlit.app`

### Requirements for Streamlit

Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing dependencies ...
    "streamlit>=1.28.0",
]
```

---

## Option 2: FastAPI REST API

### Local Testing

```bash
# Install uvicorn
poetry add uvicorn
# or
pip install uvicorn

# Run API
uvicorn api:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs

### Deploy to Railway (Easy & Free Tier)

1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login:**
   ```bash
   railway login
   ```

3. **Initialize project:**
   ```bash
   railway init
   ```

4. **Create `Procfile`:**
   ```
   web: uvicorn api:app --host 0.0.0.0 --port $PORT
   ```

5. **Deploy:**
   ```bash
   railway up
   ```

### Deploy to Render (Free Tier)

1. **Create `render.yaml`:**
   ```yaml
   services:
     - type: web
       name: immo-eliza-api
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT
   ```

2. **Push to GitHub**

3. **Go to [Render](https://render.com)**
   - New → Web Service
   - Connect GitHub repo
   - Select `render.yaml`
   - Deploy

### Deploy to Heroku

1. **Create `Procfile`:**
   ```
   web: uvicorn api:app --host 0.0.0.0 --port $PORT
   ```

2. **Create `runtime.txt`:**
   ```
   python-3.13.0
   ```

3. **Deploy:**
   ```bash
   heroku create immo-eliza-api
   git push heroku main
   ```

---

## Option 3: Docker Deployment (Recommended for Production)

Docker configurations are provided for both Streamlit and FastAPI. You can run them separately or together using docker-compose.

### Quick Start with Docker Compose (Both Services)

Run both Streamlit app and FastAPI API together:

```bash
# Build and start both services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

**Access:**
- Streamlit App: `http://localhost:8501`
- FastAPI API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

**Stop services:**
```bash
docker-compose down
```

### Streamlit App Only

**Build:**
```bash
docker build -f Dockerfile.streamlit -t immo-eliza-streamlit .
```

**Run:**
```bash
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  immo-eliza-streamlit
```

**Access:** `http://localhost:8501`

### FastAPI API Only

**Build:**
```bash
docker build -f Dockerfile.api -t immo-eliza-api .
```

**Run:**
```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  immo-eliza-api
```

**Access:** 
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

### Deploy Docker Images to Cloud

#### Option A: Docker Hub + Any Cloud Provider

```bash
# Build images
docker build -f Dockerfile.streamlit -t yourusername/immo-eliza-streamlit .
docker build -f Dockerfile.api -t yourusername/immo-eliza-api .

# Push to Docker Hub
docker push yourusername/immo-eliza-streamlit
docker push yourusername/immo-eliza-api
```

#### Option B: Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/immo-eliza-streamlit -f Dockerfile.streamlit
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/immo-eliza-api -f Dockerfile.api

# Deploy Streamlit
gcloud run deploy immo-eliza-streamlit \
  --image gcr.io/YOUR_PROJECT_ID/immo-eliza-streamlit \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501

# Deploy API
gcloud run deploy immo-eliza-api \
  --image gcr.io/YOUR_PROJECT_ID/immo-eliza-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

#### Option C: AWS ECS / Fargate

```bash
# Build and tag for ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker build -f Dockerfile.streamlit -t YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/immo-eliza-streamlit .
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/immo-eliza-streamlit

# Create ECS task definition and deploy
```

#### Option D: Azure Container Instances

```bash
# Build and push to Azure Container Registry
az acr build --registry YOUR_REGISTRY --image immo-eliza-streamlit:latest -f Dockerfile.streamlit .
az acr build --registry YOUR_REGISTRY --image immo-eliza-api:latest -f Dockerfile.api .

# Deploy
az container create \
  --resource-group YOUR_RESOURCE_GROUP \
  --name immo-eliza-streamlit \
  --image YOUR_REGISTRY.azurecr.io/immo-eliza-streamlit:latest \
  --dns-name-label immo-eliza-streamlit \
  --ports 8501
```

### Docker Compose for Production

For production, you can use docker-compose with environment variables:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
    restart: always
    networks:
      - immo-eliza-network

  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
    restart: always
    networks:
      - immo-eliza-network

networks:
  immo-eliza-network:
    driver: bridge
```

Run with:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Important Notes

1. **Models must exist:** Make sure `models/` folder contains trained models before building Docker images
2. **Volume mounting:** Models are mounted as read-only volumes in docker-compose
3. **Health checks:** Both containers include health checks
4. **Port mapping:** 
   - Streamlit: `8501`
   - FastAPI: `8000`

---

## Environment Variables

Create `.env` file (or set in deployment platform):

```env
MODELS_FOLDER=models
PREPROCESSOR_PATH=models/preprocessor.json
```

---

## File Structure for Deployment

Make sure these files exist:
```
immo-eliza-ml/
├── models/              # Trained models (must exist!)
│   ├── *.pkl
│   └── preprocessor.json
├── app.py              # Streamlit app
├── api.py              # FastAPI app
├── pyproject.toml      # Dependencies
├── Procfile            # For Heroku/Railway
└── requirements.txt    # Optional: pip install
```

---

## Quick Start Commands

### Streamlit (Local)
```bash
poetry add streamlit
streamlit run app.py
```

### FastAPI (Local)
```bash
poetry add uvicorn fastapi
uvicorn api:app --reload
```

### Test API
```bash
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

---

## Troubleshooting

### Models not found
- Make sure `models/` folder exists with trained models
- Run `python -m immo_eliza_ml.main` first

### Port already in use
- Change port: `uvicorn api:app --port 8001`
- Or kill process: `lsof -ti:8000 | xargs kill`

### CORS errors (API)
- Already configured in `api.py` with CORS middleware
- Adjust `allow_origins` if needed

---

## Recommended: Streamlit Cloud

**Easiest option for ML demos:**
- ✅ Free
- ✅ No server management
- ✅ Automatic HTTPS
- ✅ Easy GitHub integration
- ✅ Perfect for ML demos

**Steps:**
1. Push code to GitHub
2. Go to streamlit.io/cloud
3. Connect repo
4. Deploy!

Your app will be live in minutes! 🚀


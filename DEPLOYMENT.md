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

## Option 3: Docker Deployment

### Create `Dockerfile`:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run:

```bash
docker build -t immo-eliza-api .
docker run -p 8000:8000 immo-eliza-api
```

### Deploy to Docker Hub / Cloud Run:

```bash
# Build and push
docker build -t yourusername/immo-eliza-api .
docker push yourusername/immo-eliza-api

# Deploy to Google Cloud Run
gcloud run deploy immo-eliza-api \
  --image yourusername/immo-eliza-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

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


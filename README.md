# HealthAI – Integrated Patient Healthcare & AI Risk Assessment Platform

A clean, modern, full-stack healthcare management platform featuring:
- 📊 **3 AI/ML Prediction Models** – Alzheimer's, Diabetes, and Cardiovascular risk assessment
- 🏥 **Patient & Doctor Portals** – Separate views with full record management
- 📋 **Lab Report Tracking** – Upload & OCR extraction with clinical reference ranges
- 💊 **Prescription & Medicine Management** – Full CRUD for medications and schedules
- ⏰ **Medicine Reminders** – Scheduled alerts and compliance tracking
- 🤖 **AI Health Chatbot** – Scoped to patient record, with medical disclaimers
- 📈 **Longitudinal Trend Charts** – Glucose, Blood Pressure, Cholesterol over time

---

## Quick Start (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train ML models (first time only)
```bash
python -m backend.ml.train_models
```

### 3. Run the server
```bash
uvicorn backend.main:app --reload
```

Open http://localhost:8000 in your browser.

**Demo credentials:**
- Patient: `patient@health.org` / `patient123`
- Doctor: `doctor@hospital.org` / `doctor123`

---

## Quick Start (Docker)

```bash
# Build and run
docker compose up --build

# Or pull pre-built and run
docker compose up
```

Open http://localhost:8000

---

## Project Structure

```
health_ai_app/
├── backend/
│   ├── api/          # FastAPI route handlers
│   ├── ml/           # ML model training & prediction
│   ├── models/       # SQLAlchemy ORM + Pydantic schemas
│   ├── ocr/          # Lab report & prescription parsers
│   ├── services/     # Business logic services
│   └── static/       # Frontend HTML/JS/CSS
├── data/             # Training datasets
├── ml_models/        # Trained model artifacts (.joblib)
├── tests/            # Pytest test suite
├── uploads/          # Patient uploaded documents
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## API Documentation

Swagger UI available at: http://localhost:8000/docs  
ReDoc available at: http://localhost:8000/redoc

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/auth/register | Register patient/doctor |
| POST | /api/auth/login | Get JWT access token |
| GET | /api/patient/profile/{id} | Get patient profile |
| PUT | /api/patient/profile/{id} | Update profile |
| POST | /api/patient/vitals/{id} | Log new vitals |
| GET | /api/patient/trends/{id} | Trend chart data |
| POST | /api/ml/predict/alzheimer/{id} | Alzheimer risk score |
| POST | /api/ml/predict/diabetes/{id} | Diabetes risk score |
| POST | /api/ml/predict/cardiovascular/{id} | CVD risk score |
| GET | /api/ml/history/{id} | Assessment history |
| POST | /api/documents/upload-report | Upload lab report |
| POST | /api/documents/upload-prescription | Upload prescription |
| GET | /api/reminders/{id} | Get medicine reminders |
| POST | /api/reminders/{id} | Add reminder |
| PATCH | /api/reminders/{reminder_id}/status | Mark taken |
| DELETE | /api/reminders/{reminder_id} | Delete reminder |
| POST | /api/chatbot/{id}/query | AI chatbot query |

---

## Running Tests

```bash
# All tests
pytest -v

# With coverage report
pytest --cov=backend --cov-report=term-missing
```

---

## AI Models

| Model | Algorithm | Accuracy | Dataset |
|-------|-----------|----------|---------|
| Alzheimer's | Logistic Regression | 83% | 2,149 patient records |
| Diabetes | Logistic Regression | 80% | Pima Indians Dataset |
| Cardiovascular (CDC) | PCA + KNN | ~85% | CDC Heart Disease Surveillance |
| Cardiovascular (Framingham) | Clinical Score | Validated | Framingham Heart Study |

> ⚠️ **Medical Disclaimer**: All AI predictions are statistical risk estimates only. They are not medical diagnoses. Always consult a qualified healthcare professional.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| SECRET_KEY | changeme | JWT signing secret (change in production!) |
| DEBUG | false | Enable debug logging |

---

## License

For educational and research purposes only. Not for clinical use.

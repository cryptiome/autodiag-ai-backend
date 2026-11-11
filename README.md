---
title: AutoDiag AI Backend
emoji: 🚗
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
python_version: "3.12.12"
app_file: app.py
pinned: false
---

# AutoDiag AI Backend
# AutoDiag AI Backend

Production-ready Flask backend for the AutoDiag AI mobile application.

## Runtime

- Flask REST API
- TensorFlow/Keras inference model
- Firebase Authentication token verification
- Cloud Firestore diagnosis persistence
- 118 symptom inputs
- 354 encoded model inputs
- 35 fault classes
- Top-3 predictions

## API

### Health

```http
GET /health
```

### Readiness

```http
GET /ready
```

### Diagnose

```http
POST /api/diagnose
Authorization: Bearer <Firebase ID Token>
Content-Type: application/json
```

Request:

```json
{
  "symptomAnswers": {
    "S001": 1,
    "S002": 0,
    "S003": -1
  },
  "notes": "optional"
}
```

The `symptomAnswers` object must contain all IDs from `S001` through `S118`.

Allowed values:

```text
1  = Yes
0  = No
-1 = Unknown / Unasked
```

The client does not send a user ID. The backend gets the UID from the verified Firebase ID token.

## Local development

Create a local `.env` from `.env.example`.

Put the Firebase service-account file outside source control and set:

```env
FIREBASE_SERVICE_ACCOUNT_PATH=firebase-service-account.json
FIREBASE_PROJECT_ID=your-project-id
```

Install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run locally:

```powershell
python -m src.api.app
```

Default local address:

```text
http://0.0.0.0:5000
```

## Render deployment

The repository includes `render.yaml`.

Render settings for manual setup:

```text
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn src.api.wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
Health Check Path: /health
```

Required Render environment values:

```text
FIREBASE_PROJECT_ID
FIREBASE_SERVICE_ACCOUNT_JSON
```

Paste the Firebase service-account JSON as the value of `FIREBASE_SERVICE_ACCOUNT_JSON`.

Alternative: upload the JSON as a Render Secret File and set:

```text
FIREBASE_SERVICE_ACCOUNT_PATH=/etc/secrets/firebase-service-account.json
```

Do not commit `.env` or the Firebase service-account JSON to GitHub.

## Validation

Run before pushing:

```powershell
python scripts/validate_project.py
```

After the deployed service is live, open:

```text
https://YOUR-SERVICE.onrender.com/health
```

and:

```text
https://YOUR-SERVICE.onrender.com/ready
```

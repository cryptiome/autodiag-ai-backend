import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('GRADIO_SSR_MODE', 'false')

from pathlib import Path

import spaces
from fastapi import Request
from fastapi.responses import JSONResponse
from firebase_admin import auth as firebase_auth
from gradio import Server

from src.api.diagnosis_service import (
    DiagnosisService,
    FaultCatalogError,
    UserNotFoundError,
    UserSuspendedError,
)
from src.api.errors import ApiError
from src.api.firebase_service import (
    AuthorizationHeaderError,
    FirebaseService,
)
from src.api.request_validation import (
    MIN_KNOWN_ANSWERS,
    validate_diagnosis_request,
)
from src.prediction.predictor import (
    ArtifactIntegrityError,
    AutoDiagPredictor,
)

BACKEND_VERSION = '1.0.0'

ROOT = Path(__file__).resolve().parent

predictor = AutoDiagPredictor(
    project_root=ROOT,
)

firebase_service = FirebaseService(
    project_root=ROOT,
)

diagnosis_service = DiagnosisService(
    predictor,
    firebase_service,
)

app = Server()


@spaces.GPU(duration=1)
def _zerogpu_probe() -> str:
    return 'ok'


@app.api(
    name='zerogpu_probe',
    api_visibility='private',
)
def zerogpu_probe() -> str:
    return _zerogpu_probe()


def api_error_response(error):
    return JSONResponse(
        status_code=error.status_code,
        content=error.to_dict(),
    )


def auth_error_to_api(error):
    if isinstance(
        error,
        firebase_auth.RevokedIdTokenError,
    ):
        return ApiError(
            'TOKEN_REVOKED',
            'Firebase ID token has been revoked.',
            401,
        )

    if isinstance(
        error,
        firebase_auth.ExpiredIdTokenError,
    ):
        return ApiError(
            'TOKEN_EXPIRED',
            'Firebase ID token has expired.',
            401,
        )

    if isinstance(
        error,
        firebase_auth.UserDisabledError,
    ):
        return ApiError(
            'USER_DISABLED',
            'Firebase user is disabled.',
            403,
        )

    if isinstance(
        error,
        firebase_auth.InvalidIdTokenError,
    ):
        return ApiError(
            'INVALID_TOKEN',
            'Firebase ID token is invalid.',
            401,
        )

    return ApiError(
        'AUTHENTICATION_FAILED',
        'Firebase authentication failed.',
        401,
    )


@app.api(name='space_status')
def space_status() -> str:
    return 'AutoDiag AI API is running.'


@app.get('/')
async def root_endpoint():
    return {
        'service': 'autodiag-api',
        'status': 'ok',
        'backendVersion': BACKEND_VERSION,
    }


@app.get('/health')
async def health():
    metadata = predictor.metadata

    return {
        'status': 'ok',
        'service': 'autodiag-api',
        'backendVersion': BACKEND_VERSION,
        'modelVersion': metadata.get(
            'model_version'
        ),
        'datasetVersion': metadata.get(
            'dataset_version'
        ),
        'firestorePersistence': True,
        'rawSymptoms': metadata.get(
            'raw_symptoms',
            118,
        ),
        'inputDimension': metadata.get(
            'input_dimension',
            354,
        ),
        'outputClasses': metadata.get(
            'output_classes',
            35,
        ),
        'minimumKnownAnswers': MIN_KNOWN_ANSWERS,
    }


@app.get('/ready')
async def ready():
    try:
        firebase_service.assert_runtime_ready()

    except Exception as error:
        print(
            'Readiness check failed: '
            f'{type(error).__name__}: {error}',
            flush=True,
        )

        return JSONResponse(
            status_code=503,
            content={
                'status': 'not_ready',
                'service': 'autodiag-api',
                'backendVersion': BACKEND_VERSION,
            },
        )

    return {
        'status': 'ready',
        'service': 'autodiag-api',
        'backendVersion': BACKEND_VERSION,
        'modelArtifacts': 'verified',
        'firestore': 'reachable',
        'masterCatalog': 'available',
    }


@app.post('/api/diagnose')
async def diagnose(request: Request):
    try:
        auth_context = (
            firebase_service.verify_authorization_header(
                request.headers.get('Authorization')
            )
        )

    except AuthorizationHeaderError as error:
        return api_error_response(
            ApiError(
                'MISSING_OR_INVALID_AUTHORIZATION',
                str(error),
                401,
            )
        )

    except Exception as error:
        print(
            'Firebase token verification failed: '
            f'{type(error).__name__}: {error}',
            flush=True,
        )

        return api_error_response(
            auth_error_to_api(error)
        )

    content_type = (
        request.headers
        .get('content-type', '')
        .lower()
    )

    if 'application/json' not in content_type:
        return api_error_response(
            ApiError(
                'JSON_REQUIRED',
                'Content-Type must be application/json.',
                415,
            )
        )

    try:
        payload = await request.json()

    except Exception:
        return api_error_response(
            ApiError(
                'INVALID_JSON_BODY',
                'Request body must contain valid JSON.',
                400,
            )
        )

    try:
        validated = validate_diagnosis_request(
            payload,
            predictor,
        )

    except ApiError as error:
        return api_error_response(error)

    try:
        result = diagnosis_service.diagnose(
            uid=auth_context['uid'],
            symptom_answers=validated[
                'symptomAnswers'
            ],
            notes=validated['notes'],
        )

    except UserNotFoundError as error:
        return api_error_response(
            ApiError(
                'USER_PROFILE_NOT_FOUND',
                str(error),
                403,
            )
        )

    except UserSuspendedError as error:
        return api_error_response(
            ApiError(
                'USER_SUSPENDED',
                str(error),
                403,
            )
        )

    except FaultCatalogError as error:
        return api_error_response(
            ApiError(
                'FAULT_CATALOG_ERROR',
                str(error),
                500,
            )
        )

    except ArtifactIntegrityError as error:
        return api_error_response(
            ApiError(
                'MODEL_ARTIFACT_ERROR',
                str(error),
                500,
            )
        )

    except Exception as error:
        print(
            'Diagnosis failed: '
            f'{type(error).__name__}: {error}',
            flush=True,
        )

        return api_error_response(
            ApiError(
                'DIAGNOSIS_FAILED',
                'Diagnosis could not be completed.',
                500,
            )
        )

    return result


demo = app


if __name__ == '__main__':
    demo.launch(ssr_mode=False)

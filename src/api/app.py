import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from firebase_admin import auth as firebase_auth

from src.api.diagnosis_service import (
    DiagnosisService,
    FaultCatalogError,
    UserNotFoundError,
    UserSuspendedError,
)
from src.api.errors import ApiError
from src.api.firebase_service import AuthorizationHeaderError, FirebaseService
from src.api.request_validation import MIN_KNOWN_ANSWERS, validate_diagnosis_request
from src.prediction.predictor import ArtifactIntegrityError, AutoDiagPredictor

BACKEND_VERSION = '1.0.0'


def _auth_error_to_api(error):
    if isinstance(error, firebase_auth.RevokedIdTokenError):
        return ApiError('TOKEN_REVOKED', 'Firebase ID token has been revoked.', 401)
    if isinstance(error, firebase_auth.ExpiredIdTokenError):
        return ApiError('TOKEN_EXPIRED', 'Firebase ID token has expired.', 401)
    if isinstance(error, firebase_auth.UserDisabledError):
        return ApiError('USER_DISABLED', 'Firebase user is disabled.', 403)
    if isinstance(error, firebase_auth.InvalidIdTokenError):
        return ApiError('INVALID_TOKEN', 'Firebase ID token is invalid.', 401)
    return ApiError('AUTHENTICATION_FAILED', 'Firebase authentication failed.', 401)


def create_app(
    firebase_service=None,
    predictor=None,
    diagnosis_service=None,
    project_root=None,
):
    root = Path(project_root or Path(__file__).resolve().parents[2])
    load_dotenv(root / '.env')

    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    app.config['MAX_CONTENT_LENGTH'] = 256 * 1024

    predictor = predictor or AutoDiagPredictor(project_root=root)
    firebase_service = firebase_service or FirebaseService(project_root=root)
    diagnosis_service = diagnosis_service or DiagnosisService(
        predictor,
        firebase_service,
    )

    app.extensions['autodiag_predictor'] = predictor
    app.extensions['autodiag_firebase_service'] = firebase_service
    app.extensions['autodiag_diagnosis_service'] = diagnosis_service

    @app.after_request
    def add_security_headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        return response

    @app.errorhandler(ApiError)
    def handle_api_error(error):
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(413)
    def handle_payload_too_large(_error):
        error = ApiError(
            'REQUEST_TOO_LARGE',
            'Request body is too large.',
            413,
        )
        return jsonify(error.to_dict()), 413

    @app.get('/')
    def root_endpoint():
        return jsonify({
            'service': 'autodiag-api',
            'status': 'ok',
            'backendVersion': BACKEND_VERSION,
        })

    @app.get('/health')
    def health():
        metadata = predictor.metadata
        return jsonify({
            'status': 'ok',
            'service': 'autodiag-api',
            'backendVersion': BACKEND_VERSION,
            'modelVersion': metadata.get('model_version'),
            'datasetVersion': metadata.get('dataset_version'),
            'firestorePersistence': True,
            'rawSymptoms': metadata.get('raw_symptoms', 118),
            'inputDimension': metadata.get('input_dimension', 354),
            'outputClasses': metadata.get('output_classes', 35),
            'minimumKnownAnswers': MIN_KNOWN_ANSWERS,
        })

    @app.get('/ready')
    def ready():
        try:
            firebase_service.assert_runtime_ready()
        except Exception:
            return jsonify({
                'status': 'not_ready',
                'service': 'autodiag-api',
                'backendVersion': BACKEND_VERSION,
            }), 503

        return jsonify({
            'status': 'ready',
            'service': 'autodiag-api',
            'backendVersion': BACKEND_VERSION,
            'modelArtifacts': 'verified',
            'firestore': 'reachable',
            'masterCatalog': 'available',
        }), 200

    @app.post('/api/diagnose')
    def diagnose():
        try:
            auth_context = firebase_service.verify_authorization_header(
                request.headers.get('Authorization')
            )
        except AuthorizationHeaderError as error:
            raise ApiError(
                'MISSING_OR_INVALID_AUTHORIZATION',
                str(error),
                401,
            ) from error
        except Exception as error:
            raise _auth_error_to_api(error) from error

        if not request.is_json:
            raise ApiError(
                'JSON_REQUIRED',
                'Content-Type must be application/json.',
                415,
            )

        payload = request.get_json(silent=True)
        validated = validate_diagnosis_request(payload, predictor)

        try:
            result = diagnosis_service.diagnose(
                uid=auth_context['uid'],
                symptom_answers=validated['symptomAnswers'],
                notes=validated['notes'],
            )
        except UserNotFoundError as error:
            raise ApiError(
                'USER_PROFILE_NOT_FOUND',
                str(error),
                403,
            ) from error
        except UserSuspendedError as error:
            raise ApiError(
                'USER_SUSPENDED',
                str(error),
                403,
            ) from error
        except FaultCatalogError as error:
            raise ApiError(
                'FAULT_CATALOG_ERROR',
                str(error),
                500,
            ) from error
        except ArtifactIntegrityError as error:
            raise ApiError(
                'MODEL_ARTIFACT_ERROR',
                str(error),
                500,
            ) from error
        except Exception as error:
            app.logger.exception(
                'Diagnosis request failed after authentication.'
            )
            raise ApiError(
                'DIAGNOSIS_FAILED',
                'Diagnosis could not be completed.',
                500,
            ) from error

        return jsonify(result), 200

    return app


def run():
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / '.env')

    app = create_app(project_root=root)
    host = os.getenv('FLASK_HOST', '0.0.0.0').strip() or '0.0.0.0'
    port = int(os.getenv('PORT', os.getenv('FLASK_PORT', '5000')))
    debug = os.getenv('FLASK_DEBUG', 'false').strip().lower() == 'true'

    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=False,
    )


if __name__ == '__main__':
    run()

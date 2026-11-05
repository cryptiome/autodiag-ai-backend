import json
import os
from pathlib import Path

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import auth, credentials, firestore


class FirebaseConfigurationError(RuntimeError):
    pass


class AuthorizationHeaderError(ValueError):
    pass


class FirebaseService:
    def __init__(self, project_root=None):
        self.root = Path(
            project_root or Path(__file__).resolve().parents[2]
        )
        load_dotenv(self.root / '.env')
        self._initialize_admin()
        self.db = firestore.client()

    def _initialize_admin(self):
        if firebase_admin._apps:
            return

        credential = self._load_credential()

        options = {}
        project_id = os.getenv('FIREBASE_PROJECT_ID', '').strip()
        if project_id:
            options['projectId'] = project_id

        firebase_admin.initialize_app(
            credential,
            options or None,
        )

    def _load_credential(self):
        raw_json = os.getenv(
            'FIREBASE_SERVICE_ACCOUNT_JSON',
            '',
        ).strip()

        if raw_json:
            try:
                service_account = json.loads(raw_json)
            except json.JSONDecodeError as error:
                raise FirebaseConfigurationError(
                    'FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON.'
                ) from error

            if not isinstance(service_account, dict):
                raise FirebaseConfigurationError(
                    'FIREBASE_SERVICE_ACCOUNT_JSON must contain a JSON object.'
                )

            return credentials.Certificate(service_account)

        service_path = os.getenv(
            'FIREBASE_SERVICE_ACCOUNT_PATH',
            '',
        ).strip()

        if not service_path:
            raise FirebaseConfigurationError(
                'Configure FIREBASE_SERVICE_ACCOUNT_JSON or '
                'FIREBASE_SERVICE_ACCOUNT_PATH.'
            )

        key_path = Path(service_path)
        if not key_path.is_absolute():
            key_path = (self.root / key_path).resolve()

        if not key_path.exists():
            raise FirebaseConfigurationError(
                f'Firebase service account file not found: {key_path}'
            )

        return credentials.Certificate(str(key_path))

    @staticmethod
    def _extract_bearer_token(authorization_header):
        if (
            not isinstance(authorization_header, str)
            or not authorization_header.strip()
        ):
            raise AuthorizationHeaderError(
                'Missing Authorization header.'
            )

        parts = authorization_header.strip().split()

        if (
            len(parts) != 2
            or parts[0].lower() != 'bearer'
            or not parts[1].strip()
        ):
            raise AuthorizationHeaderError(
                'Authorization header must use '
                'Bearer <Firebase ID Token>.'
            )

        return parts[1].strip()

    def verify_authorization_header(self, authorization_header):
        token = self._extract_bearer_token(authorization_header)

        decoded = auth.verify_id_token(
            token,
            check_revoked=True,
        )

        uid = decoded.get('uid') or decoded.get('sub')

        if not uid:
            raise auth.InvalidIdTokenError(
                'Verified Firebase token does not contain a UID.'
            )

        return {
            'uid': str(uid),
            'claims': decoded,
        }

    def assert_application_user_active(self, uid):
        snapshot = (
            self.db.collection('users')
            .document(uid)
            .get()
        )

        if not snapshot.exists:
            raise LookupError(
                'Application user document was not found.'
            )

        data = snapshot.to_dict() or {}
        status = str(
            data.get('status', '')
        ).strip().lower()

        if status != 'active':
            raise PermissionError(
                'Application user is not active.'
            )

        return data

    def create_processing_diagnosis(
        self,
        uid,
        symptom_answers,
        notes,
    ):
        ref = self.db.collection('diagnoses').document()

        ref.set({
            'userId': uid,
            'symptomAnswers': dict(symptom_answers),
            'notes': notes,
            'predictedFaultId': None,
            'confidence': None,
            'predictions': [],
            'status': 'processing',
            'createdAt': firestore.SERVER_TIMESTAMP,
            'completedAt': None,
        })

        return ref

    def get_faults(self, fault_ids):
        refs = [
            self.db.collection('faults').document(fault_id)
            for fault_id in fault_ids
        ]

        snapshots = list(self.db.get_all(refs))
        by_id = {}

        for snapshot in snapshots:
            if not snapshot.exists:
                continue

            data = snapshot.to_dict() or {}

            if data.get('isActive') is False:
                continue

            by_id[snapshot.id] = data

        missing = [
            fault_id
            for fault_id in fault_ids
            if fault_id not in by_id
        ]

        if missing:
            raise LookupError(
                'Missing or inactive fault documents: '
                + ', '.join(missing)
            )

        return by_id

    def assert_runtime_ready(self):
        refs = [
            self.db.collection('symptoms').document('S001'),
            self.db.collection('symptoms').document('S118'),
            self.db.collection('faults').document('F001'),
            self.db.collection('faults').document('F035'),
        ]

        snapshots = list(self.db.get_all(refs))

        if (
            len(snapshots) != 4
            or any(not snapshot.exists for snapshot in snapshots)
        ):
            raise RuntimeError(
                'Required master catalog documents are unavailable.'
            )

        fault_snapshots = [
            snapshot
            for snapshot in snapshots
            if snapshot.id.startswith('F')
        ]

        if any(
            (snapshot.to_dict() or {}).get('isActive') is False
            for snapshot in fault_snapshots
        ):
            raise RuntimeError(
                'Required master catalog fault is inactive.'
            )

        return True

    @staticmethod
    def complete_diagnosis(ref, result):
        ref.update({
            'predictedFaultId': result['predictedFaultId'],
            'confidence': float(result['confidence']),
            'predictions': [
                {
                    'faultId': item['faultId'],
                    'confidence': float(item['confidence']),
                }
                for item in result['predictions']
            ],
            'status': 'completed',
            'completedAt': firestore.SERVER_TIMESTAMP,
        })

    @staticmethod
    def fail_diagnosis(ref):
        if ref is None:
            return

        ref.update({
            'status': 'failed',
            'completedAt': firestore.SERVER_TIMESTAMP,
        })

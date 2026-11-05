import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


class PredictionInputError(ValueError):
    pass


class ArtifactIntegrityError(RuntimeError):
    pass


class AutoDiagPredictor:
    ALLOWED_VALUES = {-1, 0, 1}

    def __init__(
        self,
        project_root=None,
        verify_artifacts=True,
    ):
        self.root = Path(
            project_root or Path(__file__).resolve().parents[2]
        )
        self.artifacts = self.root / 'artifacts'

        self.model_path = (
            self.artifacts / 'autodiag_model.keras'
        )
        self.feature_order_path = (
            self.artifacts / 'feature_order.json'
        )
        self.label_map_path = (
            self.artifacts / 'fault_label_map.json'
        )
        self.metadata_path = (
            self.artifacts / 'model_metadata.json'
        )
        self.manifest_path = (
            self.artifacts / 'model_manifest.json'
        )

        required = [
            self.model_path,
            self.feature_order_path,
            self.label_map_path,
            self.metadata_path,
            self.manifest_path,
        ]

        missing = [
            str(path.relative_to(self.root))
            for path in required
            if not path.exists()
        ]

        if missing:
            raise ArtifactIntegrityError(
                'Missing model artifacts: '
                + ', '.join(missing)
            )

        self.feature_order = self._load_json(
            self.feature_order_path
        )
        self.fault_to_index = self._load_json(
            self.label_map_path
        )
        self.metadata = self._load_json(
            self.metadata_path
        )
        self.manifest = self._load_json(
            self.manifest_path
        )

        self.index_to_fault = {
            int(index): fault_id
            for fault_id, index in self.fault_to_index.items()
        }

        if verify_artifacts:
            self._verify_manifest()

        self._validate_contract()

        import tensorflow as tf

        self.tf = tf
        self.model = tf.keras.models.load_model(
            self.model_path,
            compile=False,
        )

        if int(self.model.input_shape[-1]) != 354:
            raise ArtifactIntegrityError(
                'Model input dimension is not 354.'
            )

        if int(self.model.output_shape[-1]) != 35:
            raise ArtifactIntegrityError(
                'Model output class count is not 35.'
            )

    @staticmethod
    def _load_json(path):
        return json.loads(
            path.read_text(encoding='utf-8')
        )

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()

        with path.open('rb') as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b'',
            ):
                digest.update(chunk)

        return digest.hexdigest()

    def _verify_manifest(self):
        if self.manifest.get('status') != 'FROZEN':
            raise ArtifactIntegrityError(
                'Model manifest is not frozen.'
            )

        for rel, expected in self.manifest.get(
            'files_sha256',
            {},
        ).items():
            path = self.root / rel

            if not path.exists():
                raise ArtifactIntegrityError(
                    f'Model artifact missing: {rel}'
                )

            actual = self._sha256(path)

            if actual != expected:
                raise ArtifactIntegrityError(
                    f'Model artifact changed: {rel}'
                )

    def _validate_contract(self):
        expected_features = [
            f'S{i:03d}'
            for i in range(1, 119)
        ]

        if self.feature_order != expected_features:
            raise ArtifactIntegrityError(
                'Feature order must be exactly S001..S118.'
            )

        if len(self.fault_to_index) != 35:
            raise ArtifactIntegrityError(
                'Fault label map must contain 35 classes.'
            )

        if sorted(self.fault_to_index.values()) != list(range(35)):
            raise ArtifactIntegrityError(
                'Fault class indices must be contiguous '
                'from 0 to 34.'
            )

        if self.metadata.get('input_dimension') != 354:
            raise ArtifactIntegrityError(
                'Metadata input dimension must be 354.'
            )

        if self.metadata.get('output_classes') != 35:
            raise ArtifactIntegrityError(
                'Metadata output classes must be 35.'
            )

    def validate_answer_map(self, symptom_answers):
        if not isinstance(symptom_answers, dict):
            raise PredictionInputError(
                'symptomAnswers must be an object/map.'
            )

        expected = set(self.feature_order)
        received = set(symptom_answers.keys())

        missing = sorted(expected - received)
        extra = sorted(received - expected)

        if missing:
            raise PredictionInputError(
                'Missing symptom IDs: '
                + ', '.join(missing)
            )

        if extra:
            raise PredictionInputError(
                'Unknown symptom IDs: '
                + ', '.join(extra)
            )

        normalized = {}
        invalid = []

        for symptom_id in self.feature_order:
            value = symptom_answers[symptom_id]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
            ):
                invalid.append(
                    f'{symptom_id}={value!r}'
                )
                continue

            value = int(value)

            if value not in self.ALLOWED_VALUES:
                invalid.append(
                    f'{symptom_id}={value}'
                )
                continue

            normalized[symptom_id] = value

        if invalid:
            raise PredictionInputError(
                'Invalid symptom values. Allowed values '
                'are -1, 0 and 1. Invalid: '
                + ', '.join(invalid)
            )

        return normalized

    def encode_answer_map(self, symptom_answers):
        answers = self.validate_answer_map(
            symptom_answers
        )

        raw = np.asarray(
            [
                answers[symptom_id]
                for symptom_id in self.feature_order
            ],
            dtype=np.int8,
        )

        encoded = np.stack(
            [
                raw == 1,
                raw == 0,
                raw == -1,
            ],
            axis=1,
        ).astype(np.float32)

        return encoded.reshape(-1)

    def _format_prediction(
        self,
        probabilities,
        top_k,
    ):
        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if probabilities.shape != (35,):
            raise ArtifactIntegrityError(
                'Model returned unexpected probability '
                f'shape: {probabilities.shape}'
            )

        if not np.isfinite(probabilities).all():
            raise ArtifactIntegrityError(
                'Model returned non-finite probabilities.'
            )

        probability_sum = float(probabilities.sum())

        if abs(probability_sum - 1.0) > 1e-3:
            raise ArtifactIntegrityError(
                'Model probability sum is invalid: '
                f'{probability_sum}'
            )

        indices = np.argsort(
            -probabilities
        )[:top_k]

        predictions = [
            {
                'faultId': self.index_to_fault[int(index)],
                'confidence': round(
                    float(probabilities[index]),
                    8,
                ),
            }
            for index in indices
        ]

        return {
            'predictedFaultId': predictions[0]['faultId'],
            'confidence': predictions[0]['confidence'],
            'predictions': predictions,
        }

    def predict(
        self,
        symptom_answers,
        top_k=3,
    ):
        if (
            not isinstance(top_k, int)
            or top_k < 1
            or top_k > 35
        ):
            raise PredictionInputError(
                'top_k must be an integer from 1 to 35.'
            )

        encoded = self.encode_answer_map(
            symptom_answers
        ).reshape(1, -1)

        probabilities = self.model.predict(
            encoded,
            verbose=0,
        )[0]

        return self._format_prediction(
            probabilities,
            top_k,
        )

from src.api.errors import ApiError
from src.prediction.predictor import PredictionInputError


ALLOWED_REQUEST_KEYS = {'symptomAnswers', 'notes'}
MAX_NOTES_LENGTH = 2000
MIN_KNOWN_ANSWERS = 4


def validate_diagnosis_request(payload, predictor):
    if not isinstance(payload, dict):
        raise ApiError('INVALID_JSON_BODY', 'Request body must be a JSON object.', 400)

    unknown = sorted(set(payload.keys()) - ALLOWED_REQUEST_KEYS)
    if unknown:
        raise ApiError(
            'UNKNOWN_REQUEST_FIELDS',
            'Unknown request fields: ' + ', '.join(unknown),
            400,
        )

    if 'symptomAnswers' not in payload:
        raise ApiError('MISSING_SYMPTOM_ANSWERS', 'symptomAnswers is required.', 400)

    try:
        symptom_answers = predictor.validate_answer_map(payload['symptomAnswers'])
    except PredictionInputError as error:
        raise ApiError('INVALID_SYMPTOM_ANSWERS', str(error), 400) from error

    known_count = sum(1 for value in symptom_answers.values() if value != -1)
    if known_count < MIN_KNOWN_ANSWERS:
        raise ApiError(
            'INSUFFICIENT_SYMPTOM_INFORMATION',
            f'At least {MIN_KNOWN_ANSWERS} symptom answers must be known (0 or 1) before diagnosis.',
            400,
        )

    notes = payload.get('notes')
    if notes is not None:
        if not isinstance(notes, str):
            raise ApiError('INVALID_NOTES', 'notes must be a string or null.', 400)
        notes = notes.strip()
        if len(notes) > MAX_NOTES_LENGTH:
            raise ApiError(
                'NOTES_TOO_LONG',
                f'notes must not exceed {MAX_NOTES_LENGTH} characters.',
                400,
            )
        if not notes:
            notes = None

    return {
        'symptomAnswers': symptom_answers,
        'notes': notes,
    }

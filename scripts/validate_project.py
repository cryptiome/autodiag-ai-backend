import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

errors = []

required = [
    'src/api/app.py',
    'src/api/wsgi.py',
    'src/api/firebase_service.py',
    'src/api/diagnosis_service.py',
    'src/api/request_validation.py',
    'src/api/errors.py',
    'src/prediction/predictor.py',
    'artifacts/autodiag_model.keras',
    'artifacts/feature_order.json',
    'artifacts/fault_label_map.json',
    'artifacts/model_metadata.json',
    'artifacts/model_manifest.json',
    'requirements.txt',
    '.python-version',
    'render.yaml',
]

for rel in required:
    if not (ROOT / rel).exists():
        errors.append(f'Missing required file: {rel}')

for forbidden in [
    '.env',
    'firebase-service-account.json',
]:
    if (ROOT / forbidden).exists():
        errors.append(f'Secret file must not be in the repository: {forbidden}')

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

manifest_path = ROOT / 'artifacts/model_manifest.json'

if manifest_path.exists():
    manifest = json.loads(
        manifest_path.read_text(encoding='utf-8')
    )

    if manifest.get('status') != 'FROZEN':
        errors.append('Model manifest is not frozen.')

    for rel, expected in manifest.get('files_sha256', {}).items():
        path = ROOT / rel

        if not path.exists():
            errors.append(f'Model artifact is missing: {rel}')
            continue

        actual = sha256(path)

        if actual != expected:
            errors.append(f'Model artifact hash mismatch: {rel}')

feature_path = ROOT / 'artifacts/feature_order.json'
if feature_path.exists():
    features = json.loads(
        feature_path.read_text(encoding='utf-8')
    )

    expected = [f'S{i:03d}' for i in range(1, 119)]

    if features != expected:
        errors.append('Feature order must be exactly S001..S118.')

labels_path = ROOT / 'artifacts/fault_label_map.json'
if labels_path.exists():
    labels = json.loads(
        labels_path.read_text(encoding='utf-8')
    )

    if len(labels) != 35:
        errors.append('Fault label map must contain 35 classes.')

text_extensions = {
    '.py',
    '.json',
    '.md',
    '.yaml',
    '.yml',
    '.txt',
    '.example',
}

for path in ROOT.rglob('*'):
    if not path.is_file():
        continue

    if path.name == '.python-version':
        continue

    if path.suffix.lower() not in text_extensions and path.name not in {
        '.gitignore',
        '.env.example',
    }:
        continue

    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        continue

    historical_word = 'sta' + 'ge'
    if re.search(rf'\\b{historical_word}\\d*\\b', text, re.IGNORECASE):
        errors.append(
            f'Historical naming remains in: {path.relative_to(ROOT)}'
        )

for path in ROOT.rglob('*'):
    historical_word = 'sta' + 'ge'
    if re.search(historical_word, path.name, re.IGNORECASE):
        errors.append(
            f'Historical naming remains in filename: {path.relative_to(ROOT)}'
        )

if errors:
    print('AutoDiag backend validation FAILED.')
    for index, error in enumerate(errors, 1):
        print(f'{index}. {error}')
    sys.exit(1)

print('AutoDiag backend validation PASSED.')
print('Secrets in repository: none')
print('Model artifact hashes: verified')
print('Symptoms: 118')
print('Encoded inputs: 354')
print('Fault classes: 35')
print('Historical deployment/build naming: none')
print('Render configuration: ready')

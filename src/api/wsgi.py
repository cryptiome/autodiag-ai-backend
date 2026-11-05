from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')

from src.api.app import create_app

app = create_app(project_root=ROOT)
application = app

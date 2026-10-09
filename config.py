import os, secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)


def _secret_key() -> str:
    """Use $SECRET_KEY if set, otherwise create one once and persist it."""
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    path = os.path.join(INSTANCE_DIR, "secret_key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


class Config:
    SECRET_KEY = _secret_key()
    DATABASE = os.path.join(INSTANCE_DIR, "gridguard.db")
    ARTIFACT_DIR = os.path.join(BASE_DIR, "ml", "artifacts")
    DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_electricity_theft_data.csv")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024      # 10 MB upload limit
    MAX_BATCH_ROWS = 20000
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    MIN_FEEDBACK_FOR_RETRAIN = 20

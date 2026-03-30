import os

class Config:
    # 🔐 Secret Key (important for sessions)
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev_secret_key")

    # 🗄️ Database (SQLite for Render - easiest)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "database.db")

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{DB_PATH}"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "ManualMind-Agent"
    app_env: str = "development"
    log_level: str = "INFO"


settings = Settings()

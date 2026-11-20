from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://earthpulse:earthpulse@localhost:5432/earthpulse"
    mlflow_tracking_uri: str | None = None
    data_dir: str = "../data"
    config_dir: str = "../config"
    earthdata_token: str | None = None
    firms_map_key: str | None = None


settings = Settings()
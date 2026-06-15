from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    chroma_persist_dir: str = "./chroma_db"
    log_level: str = "INFO"
    # Optional — only needed when swapping in a real LLM later
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

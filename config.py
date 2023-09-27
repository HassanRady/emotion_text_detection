from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):        
    KAFKA_BOOTSTRAP_SERVERS: str
    KAFKA_CLEANED_TEXT_TOPIC: str
    KAFKA_EMOTION_TOPIC: str
    EMOTION_MODEL_URL: str
    KAFKA_EMOTION_CONSUMER_GROUP: str
    KAFKA_AUTO_OFFSET_RESET: str


@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
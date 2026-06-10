import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    # TestClient поднимает приложение в памяти без реального сервера,
    # запускает lifespan (модель загружается) и даёт обычный requests-подобный клиент
    with TestClient(app) as c:
        yield c
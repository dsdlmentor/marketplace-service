from fastapi import FastAPI

app = FastAPI(title="Marketplace satisfaction service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
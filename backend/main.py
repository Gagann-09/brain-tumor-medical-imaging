from fastapi import FastAPI

app = FastAPI(title="ARMT-GAN Platform API", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to ARMT-GAN Platform API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

import uvicorn

from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.app:app", host=host, port=port)
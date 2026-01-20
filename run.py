import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    from app.config import settings

    host = settings.HOST
    port = settings.PORT
    uvicorn.run("app.main:app", host=host, port=port)


if __name__ == "__main__":
    main()

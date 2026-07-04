"""Entry point: python -m mlfix"""
import uvicorn


def main() -> None:
    uvicorn.run(
        "mlfix.server:app",
        host="127.0.0.1",  # local only, never expose
        port=8765,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
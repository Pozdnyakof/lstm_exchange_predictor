"""Запуск веб-интерфейса трейдера на http://127.0.0.1:8000."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import uvicorn


def main() -> None:
    uvicorn.run(
        "graduate_work.web.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()

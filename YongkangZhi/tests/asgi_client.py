"""Minimal ASGI JSON client using stdlib; no httpx dependency required."""

import asyncio
import json


def request(app, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    async def execute():
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        received, messages = False, []

        async def receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        scope = dict(type="http", asgi={"version": "3.0"}, http_version="1.1", method=method,
                     scheme="http", path=path, raw_path=path.encode(), query_string=b"",
                     root_path="", server=("test", 80), client=("test", 1234),
                     headers=[(b"content-type", b"application/json")])
        await app(scope, receive, send)
        status = next(message["status"] for message in messages if message["type"] == "http.response.start")
        data = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
        return status, json.loads(data)

    return asyncio.run(execute())

"""A simulated Supabase Storage API, served in-process via httpx2.MockTransport.

Keeps buckets and objects in memory. Scripted failures (status codes or
"timeout") are returned first, in order.
"""

import json
from typing import Any

import httpx2

from app.infrastructure.storage.supabase_file_storage import SupabaseFileStorage

URL = "https://project.supabase.co"
SECRET_KEY = "sb_secret_test_123"
TIMEOUT = "timeout"


class FakeSupabase:
    def __init__(
        self, buckets: set[str] | None = None, script: list[int | str] | None = None
    ) -> None:
        self.buckets: dict[str, bool] = {name: False for name in (buckets or set())}
        self.objects: dict[tuple[str, str], tuple[bytes, str]] = {}
        self.script = list(script or [])
        self.requests: list[httpx2.Request] = []
        self.sleeps: list[float] = []

    def storage(
        self, key: str = SECRET_KEY, bucket: str = "documents", **kwargs: Any
    ) -> SupabaseFileStorage:
        return SupabaseFileStorage(
            URL,
            key,
            bucket,
            transport=httpx2.MockTransport(self._handle),
            sleep=self.sleeps.append,
            **kwargs,
        )

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.script:
            item = self.script.pop(0)
            if item == TIMEOUT:
                raise httpx2.ReadTimeout("timed out", request=request)
            return httpx2.Response(int(item), json={"message": f"simulated {item}"})

        path = request.url.path.removeprefix("/storage/v1")
        parts = path.strip("/").split("/")

        if parts[0] == "bucket" and request.method == "GET":
            name = parts[1]
            if name in self.buckets:
                return httpx2.Response(
                    200, json={"id": name, "public": self.buckets[name]}
                )
            return httpx2.Response(
                400,
                json={
                    "statusCode": "404",
                    "error": "Bucket not found",
                    "message": "Bucket not found",
                },
            )
        if parts == ["bucket"] and request.method == "POST":
            body = json.loads(request.content)
            if body["id"] in self.buckets:
                return httpx2.Response(
                    400, json={"message": "The resource already exists"}
                )
            self.buckets[body["id"]] = body["public"]
            return httpx2.Response(200, json={"name": body["id"]})
        if parts[0] == "object" and request.method == "POST":
            bucket, key = parts[1], "/".join(parts[2:])
            if bucket not in self.buckets:
                return httpx2.Response(400, json={"message": "Bucket not found"})
            if (bucket, key) in self.objects and request.headers.get(
                "x-upsert"
            ) != "true":
                return httpx2.Response(
                    400, json={"message": "The resource already exists"}
                )
            self.objects[(bucket, key)] = (
                request.content,
                request.headers["content-type"],
            )
            return httpx2.Response(200, json={"Key": f"{bucket}/{key}"})
        if parts[0] == "object" and request.method == "DELETE" and len(parts) == 2:
            bucket = parts[1]
            deleted = []
            for key in json.loads(request.content)["prefixes"]:
                if self.objects.pop((bucket, key), None) is not None:
                    deleted.append({"name": key})
            return httpx2.Response(200, json=deleted)
        return httpx2.Response(404, json={"message": "unexpected route"})

import os

from minio import Minio

_DEFAULT_BUCKET = "documents"


def get_minio_client() -> Minio:
    return Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
    )


def get_bucket_name() -> str:
    return os.environ.get("MINIO_BUCKET", _DEFAULT_BUCKET)


def ensure_bucket(client: Minio | None = None) -> None:
    client = client or get_minio_client()
    bucket = get_bucket_name()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

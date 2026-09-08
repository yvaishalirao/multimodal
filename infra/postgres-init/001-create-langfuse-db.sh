#!/bin/bash
# Provisions a second database in the same Postgres instance for Langfuse,
# so observability data lives on the same datastore as everything else
# (no separate Postgres/service is introduced for Langfuse).
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE "${LANGFUSE_DB_NAME}";
EOSQL

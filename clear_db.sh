#!/usr/bin/env bash
# clear_db.sh — Truncate all tables so a fresh seed can be run.
# Usage:
#   ./clear_db.sh                  # reads DATABASE_URL from .env
#   ./clear_db.sh "postgresql://…" # pass a connection string directly

set -euo pipefail

# ── Resolve DATABASE_URL ───────────────────────────────────────────────────────
if [ -n "${1:-}" ]; then
    DATABASE_URL="$1"
elif [ -f ".env" ]; then
    DATABASE_URL=$(grep -E '^DATABASE_URL=' .env | cut -d'=' -f2-)
else
    echo "ERROR: No DATABASE_URL found. Either pass it as an argument or create a .env file."
    exit 1
fi

if [ -z "${DATABASE_URL}" ]; then
    echo "ERROR: DATABASE_URL is empty."
    exit 1
fi

# Tables in dependency order — children first so FK constraints are not violated.
TABLES=(
    meta_eval_records
    webhook_configs
    alerts
    regression_reports
    suggestions
    evaluations
    conversations
)

# ── SQLite path ────────────────────────────────────────────────────────────────
if [[ "$DATABASE_URL" == sqlite* ]]; then
    # Extract the file path after sqlite+aiosqlite:/// or sqlite:///
    DB_FILE=$(echo "$DATABASE_URL" | sed 's|.*:///||')
    if [ ! -f "$DB_FILE" ]; then
        echo "ERROR: SQLite file not found: $DB_FILE"
        exit 1
    fi
    echo "Clearing SQLite database: $DB_FILE"
    DELETES=$(printf 'DELETE FROM %s;\n' "${TABLES[@]}")
    sqlite3 "$DB_FILE" "$DELETES"
    echo "✓ All tables cleared."
    exit 0
fi

# ── PostgreSQL path ────────────────────────────────────────────────────────────
# Strip SQLAlchemy driver suffix (+asyncpg, +psycopg2, etc.) — psql doesn't need it.
PSQL_URL=$(echo "$DATABASE_URL" | sed 's/+[a-zA-Z0-9_]*//')

# Check psql is available
if ! command -v psql &>/dev/null; then
    echo "ERROR: psql not found. Install PostgreSQL client tools:"
    echo "  brew install libpq   (macOS)"
    echo "  apt install postgresql-client   (Ubuntu/Debian)"
    exit 1
fi

TABLE_LIST=$(printf '%s,\n' "${TABLES[@]}" | sed '$ s/,$//')

echo "Clearing PostgreSQL database..."
echo "  URL: $(echo "$PSQL_URL" | sed 's/:\/\/[^@]*@/:\/\/***@/'))"  # mask credentials in output

psql "$PSQL_URL" <<SQL
TRUNCATE TABLE
    $TABLE_LIST
RESTART IDENTITY CASCADE;
SQL

echo "✓ All tables cleared. Ready to run: python seed_data.py"

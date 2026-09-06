import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "artifacts" / "legal_nodes.sqlite"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

TYPES = [
    "point",
    "clause",
    "article",
    "numeric_section",
    "roman_section",
]

for node_type in TYPES:
    print("\n" + "=" * 70)
    print(f"LEAF LENGTH DISTRIBUTION: {node_type}")
    print("=" * 70)

    rows = cur.execute("""
        SELECT
            CASE
                WHEN LENGTH(TRIM(n.raw_text)) < 50
                    THEN '01. < 50'
                WHEN LENGTH(TRIM(n.raw_text)) < 100
                    THEN '02. 50-99'
                WHEN LENGTH(TRIM(n.raw_text)) < 200
                    THEN '03. 100-199'
                WHEN LENGTH(TRIM(n.raw_text)) < 400
                    THEN '04. 200-399'
                WHEN LENGTH(TRIM(n.raw_text)) < 800
                    THEN '05. 400-799'
                ELSE '06. >= 800'
            END AS bucket,
            COUNT(*)

        FROM nodes n

        WHERE n.node_type = ?
          AND NOT EXISTS (
              SELECT 1
              FROM nodes c
              WHERE c.parent_id = n.node_id
          )

        GROUP BY bucket
        ORDER BY bucket
    """, (node_type,)).fetchall()

    total = sum(count for _, count in rows)

    for bucket, count in rows:
        pct = count / total * 100 if total else 0

        print(
            f"{bucket:<20}"
            f"{count:>12,}"
            f"   {pct:>7.2f}%"
        )


# ================================================================
# SHORT POINTS + UNIQUE PARENTS
# ================================================================

print("\n" + "=" * 70)
print("SHORT POINT GROUPING POTENTIAL")
print("=" * 70)

for threshold in [50, 100, 150, 200]:
    short_points = cur.execute("""
        SELECT COUNT(*)
        FROM nodes n
        WHERE n.node_type = 'point'
          AND LENGTH(TRIM(n.raw_text)) < ?
    """, (threshold,)).fetchone()[0]

    unique_parents = cur.execute("""
        SELECT COUNT(DISTINCT n.parent_id)
        FROM nodes n
        WHERE n.node_type = 'point'
          AND LENGTH(TRIM(n.raw_text)) < ?
          AND n.parent_id IS NOT NULL
    """, (threshold,)).fetchone()[0]

    saving = short_points - unique_parents

    print(f"\nThreshold < {threshold} chars")
    print(f"Short points   : {short_points:,}")
    print(f"Unique parents : {unique_parents:,}")
    print(f"Potential save : {saving:,} vectors")


conn.close()
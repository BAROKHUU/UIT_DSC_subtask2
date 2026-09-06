import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "artifacts" / "legal_nodes.sqlite"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# ============================================================
# 1. TOTAL NODES
# ============================================================

total_nodes = cur.execute("""
    SELECT COUNT(*)
    FROM nodes
""").fetchone()[0]

print("=" * 70)
print("TOTAL NODES")
print("=" * 70)
print(f"{total_nodes:,}")


# ============================================================
# 2. STRUCTURAL LEAF NODES
# node không có bất kỳ child nào
# ============================================================

leaf_nodes = cur.execute("""
    SELECT COUNT(*)
    FROM nodes n
    WHERE NOT EXISTS (
        SELECT 1
        FROM nodes c
        WHERE c.parent_id = n.node_id
    )
""").fetchone()[0]

print("\n" + "=" * 70)
print("STRUCTURAL LEAF NODES")
print("=" * 70)

print(f"Leaf nodes     : {leaf_nodes:,}")
print(f"Total nodes    : {total_nodes:,}")
print(f"Leaf percentage: {leaf_nodes / total_nodes * 100:.2f}%")


# ============================================================
# 3. NON-LEAF NODES
# ============================================================

non_leaf_nodes = total_nodes - leaf_nodes

print("\n" + "=" * 70)
print("NON-LEAF NODES")
print("=" * 70)

print(f"Non-leaf nodes : {non_leaf_nodes:,}")
print(f"Percentage     : {non_leaf_nodes / total_nodes * 100:.2f}%")


# ============================================================
# 4. LEAF DISTRIBUTION BY NODE TYPE
# ============================================================

leaf_distribution = cur.execute("""
    SELECT
        n.node_type,
        COUNT(*) AS count
    FROM nodes n
    WHERE NOT EXISTS (
        SELECT 1
        FROM nodes c
        WHERE c.parent_id = n.node_id
    )
    GROUP BY n.node_type
    ORDER BY count DESC
""").fetchall()

print("\n" + "=" * 70)
print("LEAF DISTRIBUTION BY NODE TYPE")
print("=" * 70)

for node_type, count in leaf_distribution:
    percentage = count / leaf_nodes * 100 if leaf_nodes else 0

    print(
        f"{node_type:<25}"
        f"{count:>12,}"
        f"   {percentage:>7.2f}%"
    )


# ============================================================
# 5. ALL NODES DISTRIBUTION
# ============================================================

all_distribution = cur.execute("""
    SELECT
        node_type,
        COUNT(*) AS count
    FROM nodes
    GROUP BY node_type
    ORDER BY count DESC
""").fetchall()

print("\n" + "=" * 70)
print("ALL NODES BY TYPE")
print("=" * 70)

for node_type, count in all_distribution:
    percentage = count / total_nodes * 100

    print(
        f"{node_type:<25}"
        f"{count:>12,}"
        f"   {percentage:>7.2f}%"
    )


conn.close()
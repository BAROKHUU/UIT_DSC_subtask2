from __future__ import annotations

import sqlite3


DEFAULT_RERANKABLE_TYPES = {
    'article',
    'clause',
    'point',
    'numeric_section',
    'roman_section',
    'numbered_item',
    'letter_item',
}


def expand_hierarchy(
    conn: sqlite3.Connection,
    seed_ids: list[int],
    hierarchy_cfg: dict,
) -> list[int]:
    """Add structural relatives while returning only real SQLite node row IDs."""
    unique_seeds = list(dict.fromkeys(int(row_id) for row_id in seed_ids))
    if not unique_seeds:
        return []
    if not bool(hierarchy_cfg.get('enabled', True)):
        return unique_seeds

    seed_top_k = int(hierarchy_cfg.get('seed_top_k', 30))
    max_candidates = int(hierarchy_cfg.get('max_candidates', 160))
    max_children = int(hierarchy_cfg.get('max_children_per_seed', 20))
    max_siblings = int(hierarchy_cfg.get('max_siblings_per_seed', 12))
    parent_hops = int(hierarchy_cfg.get('parent_hops', 1))
    if max_candidates < 1:
        raise ValueError('hierarchy.max_candidates must be at least 1')
    if min(seed_top_k, max_children, max_siblings, parent_hops) < 0:
        raise ValueError('hierarchy limits cannot be negative')

    configured_types = hierarchy_cfg.get('rerankable_types')
    allowed_types = set(configured_types or DEFAULT_RERANKABLE_TYPES)
    include_parent = bool(hierarchy_cfg.get('include_parent', True))
    include_children = bool(hierarchy_cfg.get('include_children', True))
    include_siblings = bool(hierarchy_cfg.get('include_siblings', True))

    result = unique_seeds[:max_candidates]
    seen = set(result)

    def append_rows(rows) -> None:
        for row in rows:
            row_id = int(row['row_id'])
            if row_id not in seen and str(row['node_type']) in allowed_types:
                result.append(row_id)
                seen.add(row_id)
                if len(result) >= max_candidates:
                    return

    for seed_id in unique_seeds[:seed_top_k]:
        if len(result) >= max_candidates:
            break
        seed = conn.execute(
            'SELECT row_id, node_id, parent_id, node_type FROM nodes WHERE row_id = ?',
            (seed_id,),
        ).fetchone()
        if not seed:
            continue

        if include_parent and parent_hops > 0:
            parent_id = seed['parent_id']
            for _ in range(parent_hops):
                if not parent_id or len(result) >= max_candidates:
                    break
                parent = conn.execute(
                    '''
                    SELECT row_id, node_id, parent_id, node_type
                    FROM nodes WHERE node_id = ?
                    ''',
                    (parent_id,),
                ).fetchone()
                if not parent:
                    break
                append_rows([parent])
                parent_id = parent['parent_id']

        if include_children and max_children > 0 and len(result) < max_candidates:
            children = conn.execute(
                '''
                SELECT row_id, node_id, parent_id, node_type
                FROM nodes WHERE parent_id = ?
                ORDER BY start_offset LIMIT ?
                ''',
                (seed['node_id'], max_children),
            ).fetchall()
            append_rows(children)

        if (
            include_siblings
            and max_siblings > 0
            and seed['parent_id']
            and len(result) < max_candidates
        ):
            siblings = conn.execute(
                '''
                SELECT row_id, node_id, parent_id, node_type
                FROM nodes
                WHERE parent_id = ? AND row_id != ?
                ORDER BY abs(start_offset - (
                    SELECT start_offset FROM nodes WHERE row_id = ?
                )), start_offset
                LIMIT ?
                ''',
                (seed['parent_id'], seed_id, seed_id, max_siblings),
            ).fetchall()
            append_rows(siblings)

    return result[:max_candidates]

import sqlite3

from legalqa.hierarchy import expand_hierarchy


def test_expansion_returns_real_parent_children_and_siblings():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.execute(
        '''
        CREATE TABLE nodes (
            row_id INTEGER PRIMARY KEY,
            node_id TEXT,
            parent_id TEXT,
            node_type TEXT,
            start_offset INTEGER
        )
        '''
    )
    conn.executemany(
        'INSERT INTO nodes VALUES (?, ?, ?, ?, ?)',
        [
            (1, 'article', None, 'article', 0),
            (2, 'clause-1', 'article', 'clause', 10),
            (3, 'point-a', 'clause-1', 'point', 20),
            (4, 'point-b', 'clause-1', 'point', 30),
            (5, 'clause-2', 'article', 'clause', 40),
        ],
    )

    expanded = expand_hierarchy(
        conn,
        [3],
        {
            'seed_top_k': 1,
            'max_candidates': 10,
            'parent_hops': 1,
            'max_children_per_seed': 10,
            'max_siblings_per_seed': 10,
        },
    )

    assert expanded[0] == 3
    assert 2 in expanded
    assert 4 in expanded

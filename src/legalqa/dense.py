from __future__ import annotations

import json
import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from .reranker import resolve_model_source


DIRECT_LEAF_TYPES = ('article', 'clause', 'numeric_section', 'roman_section')


@dataclass(frozen=True)
class DenseUnit:
    unit_type: str
    node_id: str | None
    parent_id: str | None
    member_node_ids: tuple[str, ...]
    dense_text: str


def _ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize('NFD', text)
    without_marks = ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn')
    return without_marks.replace('đ', 'd').replace('Đ', 'D').casefold()


def _ancestor_headings(context_text: str) -> tuple[str, str]:
    article_heading = ''
    clause_heading = ''
    for raw_line in context_text.splitlines():
        line = ' '.join(raw_line.split())
        folded = _ascii_fold(line)
        if re.match(r'^dieu\s+\d', folded):
            article_heading = line
        elif re.match(r'^khoan\s+\d', folded) or re.match(r'^\d+[.)]\s+', folded):
            clause_heading = line
    return article_heading, clause_heading


def build_contextual_dense_text(
    source_name: str,
    context_text: str,
    leaf_content: str,
) -> str:
    """Build dense text without copying bodies from structural ancestors."""
    article_heading, clause_heading = _ancestor_headings(context_text)
    parts = [f'Văn bản: {re.sub(r"[-_]+", " ", source_name).strip()}']
    if article_heading:
        parts.append(f'Điều: {article_heading}')
    if clause_heading:
        parts.append(f'Khoản: {clause_heading}')
    parts.append(f'Nội dung:\n{leaf_content.strip()}')
    return '\n'.join(parts)


def _direct_leaf_rows(
    conn: sqlite3.Connection,
    point_min_chars: int,
) -> Iterable[sqlite3.Row]:
    placeholders = ','.join('?' for _ in DIRECT_LEAF_TYPES)
    return conn.execute(
        f'''
        SELECT n.node_id, n.parent_id, n.node_type, n.source_name,
               n.context_text, n.raw_text
        FROM nodes AS n
        WHERE NOT EXISTS (
            SELECT 1 FROM nodes AS child WHERE child.parent_id = n.node_id
        )
          AND (
              n.node_type IN ({placeholders})
              OR (n.node_type = 'point' AND length(trim(n.raw_text)) >= ?)
          )
        ORDER BY n.row_id
        ''',
        (*DIRECT_LEAF_TYPES, int(point_min_chars)),
    )


def _short_point_rows(
    conn: sqlite3.Connection,
    point_min_chars: int,
) -> Iterable[sqlite3.Row]:
    return conn.execute(
        '''
        SELECT n.node_id, n.parent_id, n.node_type, n.source_name,
               n.context_text, n.raw_text
        FROM nodes AS n
        WHERE n.node_type = 'point'
          AND length(trim(n.raw_text)) < ?
          AND NOT EXISTS (
              SELECT 1 FROM nodes AS child WHERE child.parent_id = n.node_id
          )
        ORDER BY coalesce(n.parent_id, n.node_id), n.start_offset, n.row_id
        ''',
        (int(point_min_chars),),
    )


def iter_semantic_units(
    conn: sqlite3.Connection,
    point_min_chars: int = 180,
) -> Iterator[DenseUnit]:
    """Yield direct semantic leaves followed by bundles of short sibling points."""
    if point_min_chars < 1:
        raise ValueError('dense.point_min_chars must be at least 1')

    for row in _direct_leaf_rows(conn, point_min_chars):
        node_id = str(row['node_id'])
        yield DenseUnit(
            unit_type=str(row['node_type']),
            node_id=node_id,
            parent_id=row['parent_id'],
            member_node_ids=(node_id,),
            dense_text=build_contextual_dense_text(
                str(row['source_name'] or ''),
                str(row['context_text'] or ''),
                str(row['raw_text'] or ''),
            ),
        )

    current_parent: str | None = None
    group: list[sqlite3.Row] = []

    def make_bundle(rows: list[sqlite3.Row]) -> DenseUnit:
        first = rows[0]
        members = tuple(str(row['node_id']) for row in rows)
        content = '\n'.join(str(row['raw_text'] or '').strip() for row in rows)
        return DenseUnit(
            unit_type='point_bundle',
            node_id=None,
            parent_id=first['parent_id'],
            member_node_ids=members,
            dense_text=build_contextual_dense_text(
                str(first['source_name'] or ''),
                str(first['context_text'] or ''),
                content,
            ),
        )

    for row in _short_point_rows(conn, point_min_chars):
        group_key = str(row['parent_id'] or row['node_id'])
        if group and group_key != current_parent:
            yield make_bundle(group)
            group = []
        current_parent = group_key
        group.append(row)
    if group:
        yield make_bundle(group)


def _load_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            'FAISS is required for dense retrieval. Install dependencies with: '
            'python -m pip install -r requirements.txt'
        ) from exc
    return faiss


def _torch_dtype(name: str, device: str) -> torch.dtype | None:
    normalized = name.casefold()
    if normalized == 'auto':
        return torch.float16 if device.startswith('cuda') else None
    if normalized in {'float16', 'fp16'}:
        return torch.float16 if device.startswith('cuda') else torch.float32
    if normalized in {'bfloat16', 'bf16'}:
        return torch.bfloat16
    if normalized in {'float32', 'fp32'}:
        return torch.float32
    raise ValueError(f'Unsupported dense.dtype: {name}')


class SemanticEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int,
        pooling: str = 'last_token',
        dtype: str = 'auto',
        normalize: bool = True,
        query_instruction: str = '',
        trust_remote_code: bool = True,
    ) -> None:
        self.device = device
        self.max_length = int(max_length)
        self.pooling = pooling
        self.normalize = bool(normalize)
        self.query_instruction = query_instruction.strip()
        model_source = resolve_model_source(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if pooling == 'last_token':
            self.tokenizer.padding_side = 'left'

        model_kwargs = {'trust_remote_code': trust_remote_code}
        resolved_dtype = _torch_dtype(dtype, device)
        if resolved_dtype is not None:
            model_kwargs['torch_dtype'] = resolved_dtype
        self.model = AutoModel.from_pretrained(model_source, **model_kwargs)
        self.model.to(device)
        self.model.eval()

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'cls':
            return hidden[:, 0]
        if self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        if self.pooling == 'last_token':
            if bool(attention_mask[:, -1].all()):
                return hidden[:, -1]
            last_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[batch_indices, last_indices]
        raise ValueError(f'Unsupported dense.pooling: {self.pooling}')

    @torch.inference_mode()
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if is_query and self.query_instruction:
            texts = [f'Instruct: {self.query_instruction}\nQuery: {text}' for text in texts]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        vectors = self._pool(outputs.last_hidden_state, inputs['attention_mask'])
        vectors = vectors.float()
        if self.normalize:
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        return vectors.cpu().numpy().astype(np.float32, copy=False)


def _batched(items: Iterable[DenseUnit], batch_size: int) -> Iterator[list[DenseUnit]]:
    batch: list[DenseUnit] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def dense_manifest_path(index_path: str | Path) -> Path:
    path = Path(index_path)
    return path.with_name(path.name + '.meta.json')


def dense_mapping_signature(conn: sqlite3.Connection) -> str:
    import hashlib

    digest = hashlib.sha256()
    rows = conn.execute(
        '''
        SELECT dense_id, unit_type, coalesce(node_id, ''),
               coalesce(parent_id, ''), member_node_ids
        FROM dense_units ORDER BY dense_id
        '''
    )
    for row in rows:
        payload = json.dumps(list(row), ensure_ascii=False, separators=(',', ':'))
        digest.update(payload.encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def build_dense_index(
    conn: sqlite3.Connection,
    index_path: str | Path,
    model_name: str,
    device: str,
    dense_cfg: dict,
    trust_remote_code: bool = True,
) -> dict:
    faiss = _load_faiss()
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = int(dense_cfg.get('batch_size', 16))
    point_min_chars = int(dense_cfg.get('point_min_chars', 180))
    if batch_size < 1:
        raise ValueError('dense.batch_size must be at least 1')

    embedder = SemanticEmbedder(
        model_name=model_name,
        device=device,
        max_length=int(dense_cfg.get('max_length', 512)),
        pooling=str(dense_cfg.get('pooling', 'last_token')),
        dtype=str(dense_cfg.get('dtype', 'auto')),
        normalize=bool(dense_cfg.get('normalize', True)),
        query_instruction=str(dense_cfg.get('query_instruction', '')),
        trust_remote_code=trust_remote_code,
    )

    index = None
    dense_id = 0
    counts: dict[str, int] = {}
    conn.execute('DELETE FROM dense_units')
    try:
        batches = _batched(iter_semantic_units(conn, point_min_chars), batch_size)
        for batch in tqdm(batches, desc='Embedding semantic leaves'):
            vectors = embedder.encode([unit.dense_text for unit in batch])
            if index is None:
                index = faiss.IndexFlatIP(int(vectors.shape[1]))
            index.add(vectors)
            for unit in batch:
                conn.execute(
                    '''
                    INSERT INTO dense_units(
                        dense_id, unit_type, node_id, parent_id, member_node_ids
                    ) VALUES (?, ?, ?, ?, ?)
                    ''',
                    (
                        dense_id,
                        unit.unit_type,
                        unit.node_id,
                        unit.parent_id,
                        json.dumps(unit.member_node_ids, ensure_ascii=False),
                    ),
                )
                counts[unit.unit_type] = counts.get(unit.unit_type, 0) + 1
                dense_id += 1

        if index is None:
            raise RuntimeError('No semantic leaves were found; dense index was not created.')

        temp_path = index_path.with_name(index_path.name + '.tmp')
        faiss.write_index(index, str(temp_path))
        os.replace(temp_path, index_path)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    manifest = {
        'version': 1,
        'index_type': 'IndexFlatIP',
        'model_name': model_name,
        'dimension': int(index.d),
        'units': int(index.ntotal),
        'unit_type_counts': counts,
        'point_min_chars': point_min_chars,
        'max_length': int(dense_cfg.get('max_length', 512)),
        'pooling': str(dense_cfg.get('pooling', 'last_token')),
        'dtype': str(dense_cfg.get('dtype', 'auto')),
        'normalize': bool(dense_cfg.get('normalize', True)),
        'query_instruction': str(dense_cfg.get('query_instruction', '')),
        'mapping_signature': dense_mapping_signature(conn),
    }
    manifest_path = dense_manifest_path(index_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return {**manifest, 'index_path': str(index_path), 'manifest_path': str(manifest_path)}


class DenseRetriever:
    def __init__(
        self,
        index_path: str | Path,
        model_name: str,
        device: str,
        dense_cfg: dict,
        trust_remote_code: bool = True,
    ) -> None:
        faiss = _load_faiss()
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(
                f'Dense index not found: {self.index_path}. Run python scripts/build_index.py first.'
            )
        manifest_path = dense_manifest_path(self.index_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f'Dense manifest not found: {manifest_path}. Rebuild the index.'
            )
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if self.manifest.get('model_name') != model_name:
            raise ValueError(
                'Configured embedding model does not match the built FAISS index: '
                f'{model_name!r} != {self.manifest.get("model_name")!r}. Rebuild the index.'
            )
        expected_settings = {
            'max_length': int(dense_cfg.get('max_length', 512)),
            'pooling': str(dense_cfg.get('pooling', 'last_token')),
            'normalize': bool(dense_cfg.get('normalize', True)),
            'point_min_chars': int(dense_cfg.get('point_min_chars', 180)),
        }
        mismatches = {
            key: (self.manifest.get(key), value)
            for key, value in expected_settings.items()
            if self.manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f'Dense configuration differs from the built FAISS index: {mismatches}. '
                'Rebuild it with python scripts/build_index.py --dense-only.'
            )
        self.index = faiss.read_index(str(self.index_path))
        if int(self.index.ntotal) != int(self.manifest.get('units', -1)):
            raise RuntimeError('FAISS index and dense manifest have different unit counts.')
        self.embedder = SemanticEmbedder(
            model_name=model_name,
            device=device,
            max_length=int(dense_cfg.get('max_length', 512)),
            pooling=str(dense_cfg.get('pooling', 'last_token')),
            dtype=str(dense_cfg.get('dtype', 'auto')),
            normalize=bool(dense_cfg.get('normalize', True)),
            query_instruction=str(dense_cfg.get('query_instruction', '')),
            trust_remote_code=trust_remote_code,
        )

    def search(self, question: str, top_k: int) -> list[tuple[int, float]]:
        if top_k < 1 or int(self.index.ntotal) == 0:
            return []
        query_vector = self.embedder.encode([question], is_query=True)
        scores, indices = self.index.search(query_vector, min(top_k, int(self.index.ntotal)))
        return [
            (int(dense_id), float(score))
            for dense_id, score in zip(indices[0], scores[0])
            if int(dense_id) >= 0
        ]


def resolve_dense_hits(
    conn: sqlite3.Connection,
    dense_hits: list[tuple[int, float]],
) -> list[dict]:
    """Expand every direct unit/bundle to real SQLite structural row IDs."""
    if not dense_hits:
        return []
    dense_ids = [int(item[0]) for item in dense_hits]
    placeholders = ','.join('?' for _ in dense_ids)
    rows = conn.execute(
        f'SELECT * FROM dense_units WHERE dense_id IN ({placeholders})',
        dense_ids,
    ).fetchall()
    unit_map = {int(row['dense_id']): dict(row) for row in rows}

    member_ids: list[str] = []
    for dense_id in dense_ids:
        unit = unit_map.get(dense_id)
        if unit:
            member_ids.extend(str(item) for item in json.loads(unit['member_node_ids']))
    member_ids = list(dict.fromkeys(member_ids))
    if not member_ids:
        return []
    node_map: dict[str, int] = {}
    for start in range(0, len(member_ids), 500):
        chunk = member_ids[start : start + 500]
        node_placeholders = ','.join('?' for _ in chunk)
        node_rows = conn.execute(
            f'SELECT node_id, row_id FROM nodes WHERE node_id IN ({node_placeholders})',
            chunk,
        ).fetchall()
        node_map.update({str(row['node_id']): int(row['row_id']) for row in node_rows})

    resolved: list[dict] = []
    for dense_id, score in dense_hits:
        unit = unit_map.get(int(dense_id))
        if not unit:
            continue
        node_ids = [str(item) for item in json.loads(unit['member_node_ids'])]
        resolved.append(
            {
                'dense_id': int(dense_id),
                'score': float(score),
                'unit_type': str(unit['unit_type']),
                'node_row_ids': [node_map[item] for item in node_ids if item in node_map],
            }
        )
    return resolved


def dense_ranked_node_ids(resolved_hits: list[dict]) -> list[tuple[int, int]]:
    """Return (row_id, dense_rank); all points in one bundle share its rank."""
    ranked: list[tuple[int, int]] = []
    seen: set[int] = set()
    for rank, hit in enumerate(resolved_hits, start=1):
        for row_id in hit['node_row_ids']:
            row_id = int(row_id)
            if row_id not in seen:
                ranked.append((row_id, rank))
                seen.add(row_id)
    return ranked

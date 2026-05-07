from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize(value: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(value))


def levenshtein_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    costs = [[0] * cols for _ in range(rows)]
    ops = [[(0, 0, 0)] * cols for _ in range(rows)]

    for i in range(1, rows):
        costs[i][0] = i
        ops[i][0] = (0, 1, 0)
    for j in range(1, cols):
        costs[0][j] = j
        ops[0][j] = (0, 0, 1)

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                ops[i][j] = ops[i - 1][j - 1]
                continue

            sub = costs[i - 1][j - 1] + 1
            delete = costs[i - 1][j] + 1
            insert = costs[i][j - 1] + 1
            best = min(sub, delete, insert)
            costs[i][j] = best
            if best == sub:
                s, d, ins = ops[i - 1][j - 1]
                ops[i][j] = (s + 1, d, ins)
            elif best == delete:
                s, d, ins = ops[i - 1][j]
                ops[i][j] = (s, d + 1, ins)
            else:
                s, d, ins = ops[i][j - 1]
                ops[i][j] = (s, d, ins + 1)

    return ops[-1][-1]


def rate(errors: int, denominator: int) -> float:
    return errors / denominator if denominator else 0.0


def similarity(reference: str, hypothesis: str) -> float:
    return SequenceMatcher(None, normalize_text(reference), normalize_text(hypothesis)).ratio()


def extract_entities(reference: str, explicit_entities: str | None = None) -> list[str]:
    explicit = [item.strip() for item in (explicit_entities or "").split(",") if item.strip()]
    if explicit:
        return explicit

    raw_candidates = re.findall(
        r"\b(?:[A-Z][\w.-]+(?:\s+[A-Z][\w.-]+)*|[A-Za-z]*\d[\w.-]*|[\w.-]{4,})\b",
        reference,
        flags=re.UNICODE,
    )
    stop = {"this", "that", "there", "from", "with", "please", "audio", "recording"}
    entities: list[str] = []
    for candidate in raw_candidates:
        normalized = normalize_text(candidate)
        if normalized and normalized not in stop and normalized not in [normalize_text(e) for e in entities]:
            entities.append(candidate)
    return entities


@dataclass
class MetricResult:
    wer: float
    cer: float
    entity_recall: float
    entity_precision: float
    entity_f1: float
    substitutions: int
    deletions: int
    insertions: int
    missed_entities: list[str]
    extra_entity_like_terms: list[str]
    similarity: float


def evaluate(reference: str, hypothesis: str, entities: str | None = None) -> MetricResult:
    ref_words = tokenize(reference)
    hyp_words = tokenize(hypothesis)
    substitutions, deletions, insertions = levenshtein_counts(ref_words, hyp_words)

    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    c_subs, c_dels, c_ins = levenshtein_counts(ref_chars, hyp_chars)

    expected_entities = extract_entities(reference, entities)
    expected_norm = [normalize_text(entity) for entity in expected_entities]
    hypothesis_norm = normalize_text(hypothesis)
    found = [entity for entity, norm in zip(expected_entities, expected_norm) if norm and norm in hypothesis_norm]
    missed = [entity for entity in expected_entities if entity not in found]

    hyp_entity_like = extract_entities(hypothesis)
    expected_set = set(expected_norm)
    extras = [
        entity
        for entity in hyp_entity_like
        if normalize_text(entity) and normalize_text(entity) not in expected_set
    ][:8]

    precision = len(found) / (len(found) + len(extras)) if found or extras else (1.0 if not expected_entities else 0.0)
    recall = len(found) / len(expected_entities) if expected_entities else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return MetricResult(
        wer=rate(substitutions + deletions + insertions, len(ref_words)),
        cer=rate(c_subs + c_dels + c_ins, len(ref_chars)),
        entity_recall=recall,
        entity_precision=precision,
        entity_f1=f1,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        missed_entities=missed,
        extra_entity_like_terms=extras,
        similarity=similarity(reference, hypothesis),
    )

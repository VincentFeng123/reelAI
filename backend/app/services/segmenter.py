from dataclasses import dataclass
import re

from .vector_search import top_k_cosine


@dataclass
class TranscriptChunk:
    chunk_index: int
    t_start: float
    t_end: float
    text: str


@dataclass
class SegmentMatch:
    chunk_index: int
    t_start: float
    t_end: float
    text: str
    score: float
    source: str = "legacy"  # "legacy" or "topic_cut" — determines post-processing


def chunk_transcript(entries: list[dict], target_sec: int = 22, min_sec: int = 15, max_sec: int = 30) -> list[TranscriptChunk]:
    chunks: list[TranscriptChunk] = []
    if not entries:
        return chunks

    start = None
    end = None
    texts: list[str] = []
    idx = 0

    for entry in entries:
        t0 = float(entry.get("start", 0.0))
        duration = float(entry.get("duration", 0.0))
        t1 = t0 + duration
        text = (entry.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue

        if start is None:
            start = t0
            end = t1
            texts.append(text)
        else:
            current_length = (end - start) if end is not None else 0.0
            next_length = t1 - start
            if texts and next_length > max_sec and current_length >= min_sec:
                chunks.append(
                    TranscriptChunk(
                        chunk_index=idx,
                        t_start=float(start),
                        t_end=float(end),
                        text=" ".join(texts),
                    )
                )
                idx += 1
                start = t0
                end = t1
                texts = [text]
            else:
                end = t1
                texts.append(text)

        length = (end - start) if start is not None and end is not None else 0.0
        if length >= target_sec or length >= max_sec:
            if length >= min_sec:
                chunks.append(
                    TranscriptChunk(
                        chunk_index=idx,
                        t_start=float(start),
                        t_end=float(end),
                        text=" ".join(texts),
                    )
                )
                idx += 1
                start = None
                end = None
                texts = []

    if start is not None and end is not None and texts:
        length = end - start
        if length >= min_sec:
            chunks.append(TranscriptChunk(chunk_index=idx, t_start=start, t_end=end, text=" ".join(texts)))

    return chunks


def select_segments(
    concept_embedding,
    chunk_embeddings,
    chunks: list[TranscriptChunk],
    concept_terms: list[str] | None = None,
    top_k: int = 3,
) -> list[SegmentMatch]:
    matches: list[SegmentMatch] = []
    candidate_count = min(len(chunks), max(top_k * 6, 12))
    ranked = top_k_cosine(concept_embedding, chunk_embeddings, top_k=candidate_count)
    scored_rows: list[tuple[int, float]] = []
    for idx, score in ranked:
        chunk = chunks[idx]
        lexical = lexical_overlap_score(chunk.text, concept_terms or [])
        blended = float(score) + 0.2 * lexical
        if concept_terms and lexical == 0:
            blended -= 0.03
        scored_rows.append((idx, blended))
    scored_rows.sort(key=lambda row: row[1], reverse=True)

    take = min(len(scored_rows), max(top_k * 3, 6))
    for idx, score in scored_rows[:take]:
        chunk = chunks[idx]
        matches.append(
            SegmentMatch(
                chunk_index=chunk.chunk_index,
                t_start=chunk.t_start,
                t_end=chunk.t_end,
                text=chunk.text,
                score=score,
            )
        )

    merged = merge_adjacent(matches, max_total_sec=60)
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]


def merge_adjacent(matches: list[SegmentMatch], max_total_sec: int = 60) -> list[SegmentMatch]:
    if not matches:
        return []

    ordered = sorted(matches, key=lambda m: m.chunk_index)
    merged: list[SegmentMatch] = [ordered[0]]

    for current in ordered[1:]:
        prev = merged[-1]
        if current.chunk_index - prev.chunk_index == 1:
            merged_len = current.t_end - prev.t_start
            if merged_len <= max_total_sec:
                merged[-1] = SegmentMatch(
                    chunk_index=prev.chunk_index,
                    t_start=prev.t_start,
                    t_end=current.t_end,
                    text=f"{prev.text} {current.text}",
                    score=max(prev.score, current.score),
                )
                continue
        merged.append(current)

    bounded: list[SegmentMatch] = []
    for m in merged:
        adjusted_end = m.t_end
        length = adjusted_end - m.t_start
        if length < 15:
            adjusted_end = m.t_start + 15
        if adjusted_end - m.t_start > 60:
            adjusted_end = m.t_start + 60
        bounded.append(
            SegmentMatch(
                chunk_index=m.chunk_index,
                t_start=m.t_start,
                t_end=adjusted_end,
                text=m.text,
                score=m.score,
            )
        )
    return bounded


def lexical_overlap_score(text: str, concept_terms: list[str]) -> float:
    term_tokens = normalize_terms(concept_terms)
    if not term_tokens:
        return 0.0
    text_tokens = normalize_terms([text])
    if not text_tokens:
        return 0.0
    overlap = term_tokens.intersection(text_tokens)
    return min(1.0, len(overlap) / max(1, len(term_tokens)))


def _basic_stem_variants(token: str) -> list[str]:
    """Generate stem variants for better lexical recall without a heavy stemmer."""
    variants: list[str] = []
    if len(token) <= 4:
        return variants
    # Strip common suffixes in order, generating multiple variants
    # Plural forms
    if token.endswith("ies") and len(token) > 5:
        variants.append(token[:-3] + "y")  # studies → study
    if token.endswith("es") and len(token) > 4:
        variants.append(token[:-2])  # processes → process
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        variants.append(token[:-1])  # functions → function
    # Verb forms
    if token.endswith("ing") and len(token) > 5:
        variants.append(token[:-3])  # learning → learn
        if len(token) > 6 and token[-4] == token[-5]:  # running → run
            variants.append(token[:-4])
    if token.endswith("ed") and len(token) > 4:
        variants.append(token[:-2])  # explained → explain
        variants.append(token[:-1])  # used → us (filtered by length)
    if token.endswith("er") and len(token) > 4:
        variants.append(token[:-2])  # teacher → teach
    # Nominal forms
    if token.endswith("ation") and len(token) > 7:
        base = token[:-5]
        variants.append(base + "ate")  # calculation → calculate
        variants.append(base)  # calculation → calcul
    elif token.endswith("tion") and len(token) > 7:
        variants.append(token[:-4])  # reduction → reduc
    if token.endswith("ment") and len(token) > 6:
        variants.append(token[:-4])  # development → develop
    if token.endswith("ness") and len(token) > 6:
        variants.append(token[:-4])  # awareness → aware
    if token.endswith("ive") and len(token) > 5:
        variants.append(token[:-3])  # derivative → derivat
        variants.append(token[:-3] + "e")  # derivative → derivate
    if token.endswith("ity") and len(token) > 5:
        variants.append(token[:-3])  # complexity → complex
    if token.endswith("ly") and len(token) > 5:
        variants.append(token[:-2])  # quickly → quick
    return [v for v in variants if len(v) >= 3]


def normalize_terms(terms: list[str]) -> set[str]:
    tokens: set[str] = set()
    for term in terms:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-']*", term.lower()):
            if len(token) >= 3:
                tokens.add(token)
                # First-pass variants
                variants = _basic_stem_variants(token)
                for variant in variants:
                    if len(variant) >= 3:
                        tokens.add(variant)
                        # Second-pass: stem the variants too (e.g. calculations→calculation→calculate)
                        for v2 in _basic_stem_variants(variant):
                            if len(v2) >= 3:
                                tokens.add(v2)
    return tokens

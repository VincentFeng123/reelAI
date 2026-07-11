"""A sentence straddling a chunk seam must reconcile to one clean sentence (no dup punctuation)."""
from __future__ import annotations

from backend.pipeline.punctuation.service import restore_transcript_punctuation

from .conftest import TargetProvider, make_words


def _long_doc(n_sentences: int = 80, per: int = 10):
    raw, tgt = [], []
    for k in range(n_sentences):
        body = " ".join(f"{stem}{k}" for stem in
                         ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
                          "iota", "kappa"][:per])
        raw.append(body)
        tgt.append(body[0].upper() + body[1:] + ".")
    return " ".join(raw), " ".join(tgt)


def test_sentence_across_chunk_boundary_reconciles_cleanly():
    text, target = _long_doc()                 # ~800 words → forces multiple chunks
    words = make_words(text)
    r = restore_transcript_punctuation(words, provider_impl=TargetProvider(target), source="")

    assert r.metadata.chunkCount >= 2          # the seam is genuinely exercised
    assert r.status in ("complete", "complete_with_repairs")
    assert r.readableText == target            # round-trips through the seam exactly
    assert ".." not in r.readableText          # no duplicated terminal punctuation
    assert " ." not in r.readableText
    assert "  " not in r.readableText
    assert [pw.word for pw in r.words] == [w["word"] for w in words]

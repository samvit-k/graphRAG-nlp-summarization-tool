"""Match synthetic personas to ReCOVery articles by semantic similarity of their shared-dictionary topics.

Topic identity is the dictionary topic ID (IPTC Media Topics qcode); names are resolved only to build embedding
inputs and for display. Articles with no topics and personas with no usable interests score NO_MATCH_SCORE.

  match.py --persona P01 --limit 50          # score P01 against the first 50 ReCOVery articles
  match.py --calibrate --persona P01         # MIND-proxy PR-AUC for scoring variants / input types
  match.py --sample-judgments 120            # write a stratified template for manual review (full corpus only)
  match.py --evaluate-judgments              # choose the threshold on the calibration split, test on the holdout
  match.py --upsert                          # also store vectors in Pinecone and check parity
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embeddings  # noqa: E402
import labeling  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = ROOT / "user_data" / "merged_dataset"
PERSONAS_PATH = MERGED_DIR / "merged.csv"
NEWS_PATH = ROOT / "user_data" / "MIND_clean" / "news.csv"
RECOVERY_PATH = ROOT / "topic_data" / "recovery" / "recovery_news_data_cleaned.csv"
GRAPHRAG_INPUT = ROOT / "graphRAG" / "input"
MIND_LABELS = MERGED_DIR / "mind_article_labels.json"
RECOVERY_LABELS = ROOT / "topic_data" / "recovery" / "recovery_article_labels.json"
INTERESTS_PATH = MERGED_DIR / "user_interests.json"
MATCHES_PATH = MERGED_DIR / "user_article_matches.csv"
CALIBRATION_PATH = MERGED_DIR / "mind_calibration_scores.csv"
JUDGMENTS_PATH = MERGED_DIR / "match_judgments.csv"
THRESHOLD_PATH = MERGED_DIR / "match_threshold.json"

PROVISIONAL = {"threshold": 0.60, "scoring": "max", "input_type": "passage"}  # until judgments are evaluated
TARGET_PRECISION = 0.80
SEED = 42
HOLDOUT_FRACTION = 0.2  # MIND history split for the proxy
JUDGMENT_HOLDOUT_FRACTION = 0.3  # reviewed pairs reserved to test the chosen threshold
MIN_PER_CLASS = {"calibration": 15, "holdout": 8}
LEDE_WORDS = 120
TAU_GRID = np.round(np.arange(0.30, 0.951, 0.05), 2)
NO_MATCH_SCORE = -1.0  # cosine lower bound: an article without topics or a persona without interests never matches


def load_env() -> None:
    for path in (ROOT / ".env", ROOT / "graphRAG" / ".env"):
        if path.is_file():
            for line in path.read_text().splitlines():
                key, sep, value = line.partition("=")
                if sep and not key.strip().startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def calibrated_defaults() -> dict:
    return json.loads(THRESHOLD_PATH.read_text()) if THRESHOLD_PATH.is_file() else PROVISIONAL


def read_csv(path: Path) -> pd.DataFrame:
    csv.field_size_limit(sys.maxsize)
    return pd.read_csv(path, dtype=str, na_filter=False, encoding="utf-8")


def load_recovery(limit: int | None) -> pd.DataFrame:
    recovery = read_csv(RECOVERY_PATH)
    if not recovery["document_id"].is_unique:
        raise ValueError("ReCOVery document_id values are not unique")
    files = defaultdict(list)
    for path in GRAPHRAG_INPUT.glob("*.txt"):
        files[path.stem.split("-", 1)[0]].append(path.name)
    recovery = recovery[recovery["body_text"].str.strip() != ""].copy()
    if bad := [i for i in recovery["document_id"] if len(files[i]) != 1]:
        raise ValueError(f"{len(bad)} documents do not map to exactly one GraphRAG input file, e.g. {bad[:5]}")
    recovery["graphrag_title"] = recovery["document_id"].map(lambda i: files[i][0])
    recovery = recovery.sort_values("document_id", key=lambda s: s.astype(int))
    return recovery.head(limit) if limit else recovery


def load_full_matches() -> pd.DataFrame:
    """The matches CSV, refusing pilot runs: every persona must be scored against the whole corpus."""
    matches = read_csv(MATCHES_PATH).astype({"score": float})
    corpus, personas = len(load_recovery(None)), len(read_csv(PERSONAS_PATH))
    per_persona = matches.groupby("persona_id").document_id.nunique()
    if len(per_persona) != personas or (per_persona != corpus).any():
        raise SystemExit(
            f"{MATCHES_PATH.name} covers {len(per_persona)}/{personas} personas and "
            f"{per_persona.min() if len(per_persona) else 0}/{corpus} articles; run full matching (no --persona/--limit) first"
        )
    return matches


def history(persona, news: pd.DataFrame) -> tuple[list[str], list[str]]:
    ids = list(dict.fromkeys(persona.history.split()))
    return [i for i in ids if i in news.index], [i for i in ids if i not in news.index]


def impressions(persona, news: pd.DataFrame) -> tuple[list[str], list[str]]:
    pairs = [item.rsplit("-", 1) for item in persona.impressions.split()]
    return (
        [i for i, c in pairs if c == "1" and i in news.index],
        [i for i, c in pairs if c == "0" and i in news.index],
    )


def label_mind(news: pd.DataFrame, ids: list[str]) -> dict[str, list[str]]:
    """{news_id: [topic_id]} from the shared dictionary."""
    return labeling.label_articles(labeling.LabelCache(MIND_LABELS), {i: labeling.mind_text(news.loc[i]) for i in ids})


def label_recovery(recovery: pd.DataFrame) -> dict[str, list[str]]:
    """{document_id: [topic_id]} from the same dictionary, shortlist and prompt as MIND."""
    texts = {row["document_id"]: labeling.recovery_text(row) for row in recovery.to_dict("records")}
    return labeling.label_articles(labeling.LabelCache(RECOVERY_LABELS), texts)


def check_topic_ids(*assignments: dict[str, list[str]]) -> None:
    vocab = labeling.vocabulary()
    unknown = sorted({t for m in assignments for ids in m.values() for t in ids} - set(vocab.topics))
    if unknown:
        raise ValueError(f"assignments use topic IDs missing from the dictionary: {unknown[:5]}")


def profile(ids: list[str], labels: dict[str, list[str]]) -> dict[str, list[str]]:
    """Interest topic ID -> supporting history articles, ordered by support then ID."""
    sources = defaultdict(list)
    for i in ids:
        for topic_id in labels[i]:
            sources[topic_id].append(i)
    return dict(sorted(sources.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def idf_weights(article_labels: dict[str, list[str]]) -> dict[str, float]:
    df = defaultdict(int)
    for labels in article_labels.values():
        for label in set(labels):
            df[label] += 1
    return {label: math.log(len(article_labels) / count) for label, count in df.items()}


def score_article(
    labels: list[str], interests: list[str], label_vec: dict[str, np.ndarray], user_matrix: np.ndarray,
    threshold: float, idf: dict[str, float] | None = None, name=str,
) -> dict:
    """Baseline: max cosine over all (article topic, user interest) pairs. Optional: IDF-weighted mean best match.

    `labels` and `interests` are topic IDs; `name` resolves an ID to its display name.
    """
    if not labels or not len(interests):
        return {"score": NO_MATCH_SCORE, "included": False, "best_article_topic_id": "", "best_article_label": "",
                "best_user_topic_id": "", "best_user_interest": "", "matched_labels": "[]"}
    sims = np.stack([label_vec[label] for label in labels]) @ user_matrix.T
    best, best_interest = sims.max(axis=1), sims.argmax(axis=1)
    weights = np.array([idf[label] for label in labels]) if idf else None
    score = float(weights @ best / weights.sum()) if weights is not None and weights.sum() > 0 else float(best.max())
    top = int(best.argmax())
    pairs = sorted(
        ((labels[a], interests[u], float(sims[a, u])) for a, u in zip(*np.nonzero(sims >= threshold))),
        key=lambda p: (-p[2], p[0], p[1]),
    )
    best_user = interests[int(best_interest[top])]
    return {
        "score": score,
        "included": score >= threshold,
        "best_article_topic_id": labels[top],
        "best_article_label": name(labels[top]),
        "best_user_topic_id": best_user,
        "best_user_interest": name(best_user),
        "matched_labels": json.dumps([
            {"article_topic_id": a, "article_label": name(a), "user_topic_id": u, "user_interest": name(u),
             "cosine": round(c, 4)}
            for a, u, c in pairs
        ]),
    }


def topic_names(topic_ids: list[str]) -> list[str]:
    vocab = labeling.vocabulary()
    return [vocab.name(t) for t in topic_ids]


def label_vectors(topic_ids: set[str]) -> dict[str, np.ndarray]:
    """Vectors keyed by topic ID, embedded from the topic names."""
    ordered = sorted(topic_ids)
    return dict(zip(ordered, embeddings.embed(topic_names(ordered), "passage")))


def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    hits = positive[order].astype(float)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum()) if hits.sum() else float("nan")


def confusion(scores: np.ndarray, relevant: np.ndarray, tau: float) -> dict:
    tp, fp = int(((scores >= tau) & relevant).sum()), int(((scores >= tau) & ~relevant).sum())
    fn = int(((scores < tau) & relevant).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"threshold": float(tau), "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp}


def run_matching(personas, news, recovery, args) -> None:
    vocab = labeling.vocabulary()
    histories = {p.persona_id: history(p, news) for p in personas.itertuples()}
    mind_labels = label_mind(news, sorted({i for resolved, _ in histories.values() for i in resolved}))
    article_labels = label_recovery(recovery)
    check_topic_ids(mind_labels, article_labels)
    if unlabeled := sum(not ids for ids in article_labels.values()):
        print(f"{unlabeled}/{len(article_labels)} ReCOVery articles have no topics and cannot match")
    idf = idf_weights(article_labels) if args.scoring == "idf" else None
    label_vec = label_vectors({t for ids in article_labels.values() for t in ids})

    rows, interests_out = [], {}
    for persona in personas.itertuples():
        resolved, missing = histories[persona.persona_id]
        interests = profile(resolved, mind_labels)
        interest_ids = list(interests)
        if not interest_ids:
            print(f"warning: {persona.persona_id} has no usable interests; every article scores {NO_MATCH_SCORE}")
        user_matrix = embeddings.embed(topic_names(interest_ids), args.input_type)
        interests_out[persona.persona_id] = {
            "user_id": persona.user_id,
            "history_resolved": len(resolved),
            "history_missing": missing,
            "interests": [{"topic_id": k, "label": vocab.name(k), "support": len(v), "source_news_ids": v}
                          for k, v in interests.items()],
        }
        scored = [
            {"persona_id": persona.persona_id, "user_id": persona.user_id, "document_id": a.document_id,
             "graphrag_title": a.graphrag_title,
             **score_article(article_labels[a.document_id], interest_ids, label_vec, user_matrix, args.threshold,
                             idf, vocab.name)}
            for a in recovery.itertuples()
        ]
        scored.sort(key=lambda r: (-r["score"], int(r["document_id"])))
        for rank, row in enumerate(scored, 1):
            row["rank"] = rank
        rows += scored
        included = sum(r["included"] for r in scored)
        print(f"{persona.persona_id}: {len(interest_ids)} interests, {included}/{len(scored)} articles included")

    columns = ["persona_id", "user_id", "document_id", "graphrag_title", "score", "rank", "included",
               "best_article_topic_id", "best_article_label", "best_user_topic_id", "best_user_interest",
               "matched_labels"]
    matches = pd.DataFrame(rows, columns=columns).assign(
        threshold=args.threshold, scoring=args.scoring, input_type=args.input_type,
        embed_model=f"{embeddings.MODEL}@{embeddings.DIMENSION}",
        label_model=f"{labeling.MODEL}/{labeling.PROMPT_VERSION}/{labeling.SHORTLIST_VERSION}",
        topic_dictionary=f"{vocab.source} {vocab.version} sha256:{vocab.sha256[:12]}",
    )
    matches.round({"score": 6}).to_csv(MATCHES_PATH, index=False)
    INTERESTS_PATH.write_text(json.dumps(interests_out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {MATCHES_PATH.name} ({len(matches):,} pairs) and {INTERESTS_PATH.name}")

    counts = pd.DataFrame(
        {tau: matches.groupby("persona_id")["score"].apply(lambda s: int((s >= tau).sum())) for tau in TAU_GRID}
    )
    print("\nincluded articles per persona by threshold:\n" + counts.to_string())

    if args.upsert:
        upsert_and_check(interests_out, article_labels, label_vec, args.input_type)


def upsert_and_check(interests_out, article_labels, label_vec, input_type) -> None:
    """Vector IDs are `<document_id>:<topic_id>` / `<persona_id>:<topic_id>`. Before the first upsert under this
    scheme, clear vectors left in these namespaces by the old `<id>:<label digest>` scheme (see the PR notes)."""
    vocab = labeling.vocabulary()
    keys = [(doc, t) for doc, ids in article_labels.items() for t in ids]
    if not keys:
        print("no ReCOVery article has topics; nothing to upsert")
        return
    article_ids = [f"{doc}:{t}" for doc, t in keys]
    article_matrix = np.stack([label_vec[t] for _, t in keys])
    embeddings.upsert("recovery-labels", article_ids, article_matrix,
                      [{"document_id": d, "topic_id": t, "label": vocab.name(t)} for d, t in keys])
    overlaps = []
    for persona_id, entry in interests_out.items():
        interests = entry["interests"]
        if not interests:
            print(f"{persona_id}: no interests to upsert")
            continue
        vectors = embeddings.embed([i["label"] for i in interests], input_type)
        embeddings.upsert(
            "user-interests",
            [f"{persona_id}:{i['topic_id']}" for i in interests],
            vectors,
            [{"persona_id": persona_id, "topic_id": i["topic_id"], "label": i["label"], "support": i["support"]}
             for i in interests],
        )
        for vector in vectors[:5]:
            local = {article_ids[j] for j in np.argsort(-(article_matrix @ vector), kind="stable")[:10]}
            remote = {i for i, _ in embeddings.query("recovery-labels", vector, 10)}
            overlaps.append(len(local & remote) / 10)
    if overlaps:
        print(f"Pinecone parity (top-10 overlap with exact local search): mean {np.mean(overlaps):.3f}, min {min(overlaps):.1f}")


def calibrate(personas, news, args) -> None:
    """MIND proxy: held-out history + clicks are positives; unclicked impressions are weak negatives only."""
    plans, needed = {}, set()
    for persona in personas.itertuples():
        resolved, _ = history(persona, news)
        rng = random.Random(f"{SEED}-{persona.persona_id}")
        shuffled = rng.sample(sorted(resolved), len(resolved))
        cut = max(1, round(len(shuffled) * HOLDOUT_FRACTION))
        train, held_out = shuffled[cut:], shuffled[:cut]
        clicked, unclicked = impressions(persona, news)
        positives = list(dict.fromkeys(held_out + [i for i in clicked if i not in train]))
        weak = [i for i in dict.fromkeys(unclicked) if i not in resolved and i not in positives]
        seen = set(resolved) | set(clicked) | set(unclicked)
        pool = sorted(i for i in news.index if i not in seen)
        random_neg = rng.sample(pool, 2 * len(positives))
        plans[persona.persona_id] = (train, positives, weak, random_neg)
        needed |= set(train) | set(positives) | set(weak) | set(random_neg)

    mind_labels = label_mind(news, sorted(needed))
    check_topic_ids(mind_labels)
    idf = idf_weights(mind_labels)
    label_vec = label_vectors({t for ids in mind_labels.values() for t in ids})

    rows = []
    for input_type in ("passage", "query"):
        for scoring in ("max", "idf"):
            for persona_id, (train, positives, weak, random_neg) in plans.items():
                interests = list(profile(train, mind_labels))  # may be empty: those pairs score NO_MATCH_SCORE
                user_matrix = embeddings.embed(topic_names(interests), input_type)
                for kind, ids in (("positive", positives), ("weak_negative", weak), ("random_negative", random_neg)):
                    for i in ids:
                        result = score_article(mind_labels[i], interests, label_vec, user_matrix, 1.0,
                                               idf if scoring == "idf" else None)
                        rows.append({"input_type": input_type, "scoring": scoring, "persona_id": persona_id,
                                     "news_id": i, "kind": kind, "score": result["score"]})
    scores = pd.DataFrame(rows)
    scores.round({"score": 6}).to_csv(CALIBRATION_PATH, index=False)

    print("PR-AUC (average precision) on the MIND proxy; random negatives are the reliable comparison:")
    for (input_type, scoring), group in scores.groupby(["input_type", "scoring"]):
        pos = group[group.kind == "positive"]
        summary = []
        for negative in ("random_negative", "weak_negative"):
            subset = pd.concat([pos, group[group.kind == negative]])
            summary.append(f"{negative}={average_precision(subset.score.to_numpy(), (subset.kind == 'positive').to_numpy()):.3f}")
        print(f"  {input_type:>7} / {scoring:<3}  " + "  ".join(summary))

    best = scores[(scores.input_type == args.input_type) & (scores.scoring == args.scoring)]
    print(f"\nthreshold sweep ({args.input_type}/{args.scoring}; precision vs random negatives):")
    for tau in TAU_GRID:
        pos, neg = best[best.kind == "positive"], best[best.kind == "random_negative"]
        tp, fp = int((pos.score >= tau).sum()), int((neg.score >= tau).sum())
        weak_rate = (best[best.kind == "weak_negative"].score >= tau).mean()
        print(f"  tau={tau:.2f}  recall={tp / len(pos):.2f}  precision={tp / max(tp + fp, 1):.2f}  weak-neg pass rate={weak_rate:.2f}")


def sample_judgments(n: int) -> None:
    """Exactly n (persona, article) pairs, split evenly across personas, then across each persona's
    score deciles (seeded remainders at both levels); scores are hidden."""
    if JUDGMENTS_PATH.exists():
        raise SystemExit(f"{JUDGMENTS_PATH} exists; refusing to overwrite reviewed judgments")
    matches = load_full_matches()
    interests = json.loads(INTERESTS_PATH.read_text(encoding="utf-8"))
    recovery = read_csv(RECOVERY_PATH).set_index("document_id")
    news_titles = read_csv(NEWS_PATH).set_index("news_id")["title"]

    rng_alloc = random.Random(SEED)

    def allocate(total: int, parts: int) -> list[int]:
        base, extra = divmod(total, parts)
        bonus = set(rng_alloc.sample(range(parts), extra))
        return [base + (k in bonus) for k in range(parts)]

    groups = [group for _, group in matches.groupby("persona_id")]
    picks = []
    for group, quota in zip(groups, allocate(n, len(groups))):
        deciles = [d for _, d in group.groupby(pd.qcut(group.score.rank(method="first"), 10, labels=False))]
        picks += [d.sample(k, random_state=SEED) for d, k in zip(deciles, allocate(quota, len(deciles))) if k]
    sample = pd.concat(picks).sample(frac=1, random_state=SEED)  # shuffled so reviewers don't see score order
    if len(sample) != n:
        raise ValueError(f"sampled {len(sample)} pairs, expected {n}")
    rng = np.random.default_rng(SEED)
    split = np.where(rng.random(len(sample)) < JUDGMENT_HOLDOUT_FRACTION, "holdout", "calibration")

    def reader(persona_id: str) -> pd.Series:
        entry = interests[persona_id]["interests"]
        sources = list(dict.fromkeys(i for item in entry for i in item["source_news_ids"]))
        return pd.Series({
            "user_interests": "; ".join(f"{i['label']} ({i['support']})" for i in entry[:20]),
            "history_titles": " | ".join(news_titles[i] for i in sources[:10]),
        })

    readers = {p: reader(p) for p in sample.persona_id.unique()}
    sample = sample.assign(
        split=split,
        title=sample.document_id.map(recovery["title"]),
        lede=sample.document_id.map(recovery["body_text"].map(lambda b: " ".join(b.split()[:LEDE_WORDS]))),
        user_interests=sample.persona_id.map(lambda p: readers[p]["user_interests"]),
        history_titles=sample.persona_id.map(lambda p: readers[p]["history_titles"]),
        relevant="",
    )[["persona_id", "document_id", "split", "title", "lede", "user_interests", "history_titles", "relevant"]]
    sample.to_csv(JUDGMENTS_PATH, index=False)
    print(f"wrote {len(sample)} pairs ({(split == 'holdout').sum()} holdout) to "
          f"{JUDGMENTS_PATH.name}; fill `relevant` with 1/0")


def evaluate_judgments() -> None:
    """Choose the threshold on the calibration split, report it on the untouched holdout, and save it."""
    judged = read_csv(JUDGMENTS_PATH)
    judged = judged[judged.relevant.isin(["0", "1"])]
    matches = load_full_matches()
    merged = judged.merge(matches, on=["persona_id", "document_id"], how="left", validate="one_to_one")
    if merged.score.isna().any():
        raise ValueError("judged pairs missing from the current matches CSV")
    for name, minimum in MIN_PER_CLASS.items():
        counts = merged[merged.split == name].relevant.value_counts()
        if counts.get("1", 0) < minimum or counts.get("0", 0) < minimum:
            raise SystemExit(f"{name} split needs >= {minimum} relevant and irrelevant judgments, has {counts.to_dict()}")

    calib, holdout = merged[merged.split == "calibration"], merged[merged.split == "holdout"]
    s, y = calib.score.to_numpy(), calib.relevant.eq("1").to_numpy()
    print(f"calibration: {len(calib)} pairs, {y.sum()} relevant; PR-AUC {average_precision(s, y):.3f}")
    sweep = [confusion(s, y, tau) for tau in TAU_GRID]
    for row in sweep:
        print(f"  tau={row['threshold']:.2f}  precision={row['precision']:.2f}  recall={row['recall']:.2f}  F1={row['f1']:.2f}")
    by_precision = [row for row in sweep if row["precision"] >= TARGET_PRECISION and row["tp"]]
    chosen, method = (by_precision[0], f"lowest threshold with precision >= {TARGET_PRECISION}") if by_precision \
        else (max(sweep, key=lambda row: row["f1"]), "F1 maximum")

    hs, hy = holdout.score.to_numpy(), holdout.relevant.eq("1").to_numpy()
    test = confusion(hs, hy, chosen["threshold"])
    print(f"holdout: {len(holdout)} pairs at tau={chosen['threshold']:.2f}: precision={test['precision']:.2f} "
          f"recall={test['recall']:.2f} F1={test['f1']:.2f}; PR-AUC {average_precision(hs, hy):.3f}")

    saved = {
        "threshold": chosen["threshold"],
        "scoring": matches.scoring.iloc[0],
        "input_type": matches.input_type.iloc[0],
        "method": f"{method} on calibration split",
        "calibration": {k: round(chosen[k], 4) for k in ("precision", "recall", "f1")} | {"pairs": int(len(calib))},
        "holdout": {k: round(test[k], 4) for k in ("precision", "recall", "f1")} | {"pairs": int(len(holdout))},
    }
    THRESHOLD_PATH.write_text(json.dumps(saved, indent=1) + "\n")
    print(f"saved to {THRESHOLD_PATH.name}; rerun matching to apply it")


def main() -> int:
    defaults = calibrated_defaults()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persona", action="append", help="persona_id to include (repeatable; default all)")
    parser.add_argument("--limit", type=int, help="score only the first N ReCOVery articles")
    parser.add_argument("--threshold", type=float, default=defaults["threshold"])
    parser.add_argument("--scoring", choices=["max", "idf"], default=defaults["scoring"])
    parser.add_argument("--input-type", choices=["passage", "query"], default=defaults["input_type"])
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--sample-judgments", type=int, metavar="N")
    parser.add_argument("--evaluate-judgments", action="store_true")
    parser.add_argument("--upsert", action="store_true")
    args = parser.parse_args()

    if args.sample_judgments:
        sample_judgments(args.sample_judgments)
        return 0
    if args.evaluate_judgments:
        evaluate_judgments()
        return 0

    load_env()
    personas = read_csv(PERSONAS_PATH)
    if args.persona:
        unknown = set(args.persona) - set(personas.persona_id)
        if unknown:
            raise SystemExit(f"unknown persona ids: {sorted(unknown)}")
        personas = personas[personas.persona_id.isin(args.persona)]
    news = read_csv(NEWS_PATH).set_index("news_id")

    if args.calibrate:
        calibrate(personas, news, args)
    else:
        run_matching(personas, news, load_recovery(args.limit), args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
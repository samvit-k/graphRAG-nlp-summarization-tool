"""Offline tests for the matching pipeline; LLM and Pinecone calls are replaced with deterministic fakes."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "matching"))
import embeddings  # noqa: E402
import labeling  # noqa: E402
import match  # noqa: E402


def unit(*values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def fake_vector(text: str) -> np.ndarray:
    seed = int(hashlib.sha1(text.casefold().encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).standard_normal(embeddings.DIMENSION).astype(np.float32)
    return v / np.linalg.norm(v)


# --- topic dictionary ---------------------------------------------------------

def test_shared_dictionary_matches_its_source_record():
    vocab = labeling.load_dictionary()  # raises if the dictionary was edited without rebuilding
    assert vocab.source == "IPTC Media Topics" and len(vocab.topics) > 1000
    assert all(t.startswith("medtop:") for t in vocab.topics)
    assert vocab.name("medtop:20000449") == "epidemic and pandemic"


def test_plural_folding_shares_search_tokens():
    assert labeling.tokens("movies") == labeling.tokens("movie")
    assert labeling.tokens("companies") == labeling.tokens("company")


# --- labeling (fixture dictionary, mocked model) --------------------------------

TOPICS = {
    "t:health": {"name": "health", "definition": "All aspects of health"},
    "t:sport": {"name": "sport", "definition": "Competitive sports"},
    "t:disease": {"name": "disease and condition", "definition": "Illness and medical conditions", "parent": "t:health"},
    "t:pandemic": {"name": "epidemic and pandemic", "definition": "Spread of a disease across a region",
                   "parent": "t:disease", "aliases": ["coronavirus", "COVID-19"]},
    "t:vaccine": {"name": "vaccine", "definition": "Preparations that give immunity to a disease", "parent": "t:disease"},
    "t:pro": {"name": "professional league", "definition": "Organised professional competition", "parent": "t:sport"},
    "t:hockey": {"name": "ice hockey", "definition": "Team sport played on ice with sticks and a puck",
                 "parent": "t:pro", "aliases": ["NHL"]},
}


@pytest.fixture
def vocab():
    return labeling.Vocabulary(TOPICS, version="test-1")


@pytest.fixture
def model(monkeypatch):
    """Replace the API call with scripted responses; records the IDs offered for each article."""
    monkeypatch.setattr(labeling, "make_client", lambda: None)
    state = SimpleNamespace(responses={}, prompts=[])

    def fake_request(_client, system, user):
        state.prompts.append((system, user))
        for needle, response in state.responses.items():
            if needle in user:
                if isinstance(response, Exception):
                    raise response
                return response
        return []

    monkeypatch.setattr(labeling, "request_topics", fake_request)
    return state


MIND_COVID = labeling.mind_text({"title": "NHL suspends season over coronavirus", "abstract": "Games halted.",
                                 "category": "sports", "subcategory": "icehockey_nhl"})
RECOVERY_COVID = labeling.recovery_text({"title": "NHL pauses play as coronavirus spreads", "body_text": "The league said..."})


def test_shortlist_is_deterministic_and_always_offers_backbone(vocab):
    shortlist = labeling.Shortlist(vocab)
    assert shortlist.backbone == ["t:disease", "t:health", "t:pro", "t:sport"]
    offered = shortlist.offered(RECOVERY_COVID)
    assert offered[:4] == shortlist.backbone and {"t:hockey", "t:pandemic"} <= set(offered)
    assert offered == labeling.Shortlist(vocab).offered(RECOVERY_COVID)
    assert shortlist.offered("Title: unrelated gardening tips") == shortlist.backbone


def test_both_datasets_get_the_same_topic_ids(tmp_path, vocab, model):
    model.responses = {"NHL": ["t:hockey", "t:pandemic"]}
    mind = labeling.label_articles(labeling.LabelCache(tmp_path / "mind.json", vocab), {"N1": MIND_COVID})
    recovery = labeling.label_articles(labeling.LabelCache(tmp_path / "rec.json", vocab), {"7": RECOVERY_COVID})
    assert mind == {"N1": ["t:hockey", "t:pandemic"]} and recovery == {"7": ["t:hockey", "t:pandemic"]}
    systems = {system for system, _ in model.prompts}
    assert len(systems) == 1  # one shared prompt and backbone for both datasets
    assert all("t:hockey | ice hockey" in user for _, user in model.prompts)


def test_validate_dedupes_and_rejects_unoffered_or_too_many():
    offered = ["a", "b", "c", "d", "e", "f"]
    assert labeling.validate([" a", "b", "a"], offered) == ["a", "b"]
    assert labeling.validate([], offered) == []
    with pytest.raises(labeling.InvalidResponse, match="not offered"):
        labeling.validate(["a", "zzz"], offered)
    with pytest.raises(labeling.InvalidResponse, match="max is 5"):
        labeling.validate(offered, offered)


def test_unknown_ids_fail_the_article_and_are_retried(tmp_path, vocab, model):
    path = tmp_path / "labels.json"
    model.responses = {"NHL": ["t:hockey", "t:made-up"]}
    with pytest.raises(RuntimeError, match="1 articles failed"):
        labeling.label_articles(labeling.LabelCache(path, vocab), {"7": RECOVERY_COVID})
    assert json.loads(path.read_text())["assignments"] == {}  # nothing cached for the bad response
    model.responses = {"NHL": ["t:vaccine"]}  # a real ID that was not offered for this text is also rejected
    with pytest.raises(RuntimeError, match="not offered"):
        labeling.label_articles(labeling.LabelCache(path, vocab), {"7": RECOVERY_COVID})
    model.responses = {"NHL": ["t:hockey"]}
    assert labeling.label_articles(labeling.LabelCache(path, vocab), {"7": RECOVERY_COVID}) == {"7": ["t:hockey"]}


def test_empty_result_is_cached_and_not_requested_again(tmp_path, vocab, model):
    path = tmp_path / "labels.json"
    text = "Title: A quiet afternoon\n\nNothing in particular happened."
    assert labeling.label_articles(labeling.LabelCache(path, vocab), {"9": text}) == {"9": []}
    assert json.loads(path.read_text())["assignments"] == {"9": []}
    model.prompts.clear()
    assert labeling.label_articles(labeling.LabelCache(path, vocab), {"9": text}) == {"9": []}
    assert model.prompts == []


def test_stale_text_is_relabeled_and_progress_survives_failure(tmp_path, vocab, model):
    path = tmp_path / "labels.json"
    model.responses = {"boom": RuntimeError("api down"), "NHL": ["t:hockey"]}
    with pytest.raises(RuntimeError, match="1 articles failed"):
        labeling.label_articles(labeling.LabelCache(path, vocab), {"a": RECOVERY_COVID, "b": "Title: boom"})
    assert set(json.loads(path.read_text())["assignments"]) == {"a"}
    model.prompts.clear()
    labeling.label_articles(labeling.LabelCache(path, vocab), {"a": RECOVERY_COVID + " changed"})
    assert len(model.prompts) == 1


def test_dictionary_change_invalidates_cached_assignments(tmp_path, vocab, model):
    path = tmp_path / "labels.json"
    model.responses = {"NHL": ["t:hockey"]}
    labeling.label_articles(labeling.LabelCache(path, vocab), {"7": RECOVERY_COVID})
    edited = json.loads(json.dumps(TOPICS))
    edited["t:hockey"]["definition"] = "Team sport on ice"
    changed = labeling.Vocabulary(edited, version="test-1")  # same version label, different content
    assert changed.sha256 != vocab.sha256
    with pytest.raises(SystemExit, match="Delete it to relabel"):
        labeling.LabelCache(path, changed)
    labeling.LabelCache(path, vocab)  # the unchanged dictionary still accepts it


def test_old_free_form_cache_is_refused(tmp_path, vocab):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"model": labeling.MODEL, "prompt_version": "v1",
                                "labels": {"N1": {"sha": "x", "labels": ["Vaccine Development"]}}}))
    with pytest.raises(SystemExit):
        labeling.LabelCache(path, vocab)


def test_cached_id_missing_from_dictionary_is_rejected(tmp_path, vocab, model):
    path = tmp_path / "labels.json"
    labeling.label_articles(labeling.LabelCache(path, vocab), {"9": "Title: quiet"})
    stored = json.loads(path.read_text())
    stored["assignments"]["9"] = ["t:gone"]
    path.write_text(json.dumps(stored))
    with pytest.raises(SystemExit, match="missing from the dictionary"):
        labeling.LabelCache(path, vocab)


# --- embeddings ---------------------------------------------------------------

def test_embedding_validation_rejects_bad_responses():
    good = SimpleNamespace(values=[1.0] * embeddings.DIMENSION)
    assert len(embeddings._validated(["x"], [good])) == 1
    for response in ([], [SimpleNamespace(values=[1.0] * 3)], [SimpleNamespace(values=[0.0] * embeddings.DIMENSION)],
                     [SimpleNamespace(values=[float("nan")] * embeddings.DIMENSION)]):
        with pytest.raises(ValueError):
            embeddings._validated(["x"], response)


def test_upsert_rejects_mismatched_or_duplicate_records():
    vectors = np.zeros((2, embeddings.DIMENSION), dtype=np.float32)
    with pytest.raises(ValueError, match="length mismatch"):
        embeddings.upsert("ns", ["a"], vectors, [{}, {}])
    with pytest.raises(ValueError, match="not unique"):
        embeddings.upsert("ns", ["a", "a"], vectors, [{}, {}])


# --- scoring ------------------------------------------------------------------

LABEL_VEC = {"A": unit(1, 0, 0), "B": unit(0, 1, 0), "C": unit(1, 1, 0)}
USERS = np.stack([unit(1, 0, 0), unit(0, 0, 1)])


def test_max_score_and_explanation():
    result = match.score_article(["B", "C"], ["u1", "u2"], LABEL_VEC, USERS, threshold=0.5)
    assert result["score"] == pytest.approx(1 / np.sqrt(2))
    assert result["included"] and result["best_article_label"] == "C" and result["best_user_interest"] == "u1"
    assert [p["article_label"] for p in json.loads(result["matched_labels"])] == ["C"]


def test_idf_variant_weights_distinctive_labels():
    result = match.score_article(["A", "B"], ["u1", "u2"], LABEL_VEC, USERS, 0.5, idf={"A": 3.0, "B": 1.0})
    assert result["score"] == pytest.approx(0.75)


def test_threshold_uses_unrounded_score():
    almost = {"X": unit(1, 1e-3, 0)}
    raw = float(almost["X"] @ USERS[0])
    assert not match.score_article(["X"], ["u1", "u2"], almost, USERS, threshold=raw + 1e-9)["included"]
    assert match.score_article(["X"], ["u1", "u2"], almost, USERS, threshold=raw)["included"]


def test_inclusion_is_monotonic_in_threshold():
    rng = np.random.default_rng(0)
    vecs = {f"L{i}": v / np.linalg.norm(v) for i, v in enumerate(rng.standard_normal((40, 8)))}
    users = rng.standard_normal((5, 8))
    users /= np.linalg.norm(users, axis=1, keepdims=True)
    articles = [[f"L{i}", f"L{i + 20}"] for i in range(20)]
    counts = [sum(match.score_article(a, list("abcde"), vecs, users, t)["included"] for a in articles) for t in match.TAU_GRID]
    assert counts == sorted(counts, reverse=True)


def test_average_precision():
    assert match.average_precision(np.array([0.9, 0.8, 0.1]), np.array([True, False, True])) == pytest.approx((1 + 2 / 3) / 2)


def test_no_topics_or_no_interests_never_match():
    for labels, users in ((["A"], np.zeros((0, 3))), ([], USERS)):
        result = match.score_article(labels, [] if not len(users) else ["u1", "u2"], LABEL_VEC, users, threshold=0.0)
        assert result["score"] == match.NO_MATCH_SCORE and not result["included"]
        assert result["matched_labels"] == "[]" and result["best_article_topic_id"] == ""


def test_explanation_resolves_ids_to_names():
    result = match.score_article(["C"], ["u1", "u2"], LABEL_VEC, USERS, 0.5, name=lambda t: f"name of {t}")
    assert result["best_article_topic_id"] == "C" and result["best_article_label"] == "name of C"
    assert json.loads(result["matched_labels"])[0]["user_interest"] == "name of u1"


def test_profile_counts_support():
    labels = {"n1": ["NBA", "Trade Deals"], "n2": ["NBA"]}
    assert match.profile(["n1", "n2"], labels) == {"NBA": ["n1", "n2"], "Trade Deals": ["n1"]}


# --- data integrity -----------------------------------------------------------

def test_recovery_documents_map_one_to_one_to_graphrag_inputs():
    recovery = match.load_recovery(None)
    assert recovery.document_id.is_unique and recovery.graphrag_title.is_unique
    assert len(recovery) == len(list(match.GRAPHRAG_INPUT.glob("*.txt")))


def test_persona_histories_resolve():
    personas, news = match.read_csv(match.PERSONAS_PATH), match.read_csv(match.NEWS_PATH).set_index("news_id")
    for persona in personas.itertuples():
        resolved, missing = match.history(persona, news)
        assert len(resolved) >= 0.9 * (len(resolved) + len(missing))  # merge.py's qualification criterion


# --- end to end: reproducibility, outputs, judgment sampling -------------------

TOPIC_IDS = sorted(labeling.vocabulary().topics)


def fake_topics(*keys: str) -> list[str]:
    """Deterministic real dictionary IDs for a fake labeler; ties MIND and ReCOVery to the same ID space."""
    return list(dict.fromkeys(TOPIC_IDS[int(hashlib.sha1(k.casefold().encode()).hexdigest()[:8], 16) % len(TOPIC_IDS)]
                              for k in keys))


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """Redirect outputs to tmp and replace LLM topics / embeddings with deterministic fakes.

    Every 7th ReCOVery article gets no topics, to exercise the zero-label path."""
    for name in ("MIND_LABELS", "RECOVERY_LABELS", "INTERESTS_PATH", "MATCHES_PATH", "JUDGMENTS_PATH", "THRESHOLD_PATH"):
        monkeypatch.setattr(match, name, tmp_path / getattr(match, name).name)
    monkeypatch.setattr(match, "label_mind", lambda news, ids: {i: fake_topics(news.loc[i, "subcategory"], news.loc[i, "category"]) for i in ids})
    monkeypatch.setattr(match, "label_recovery", lambda rec: {r.document_id: [] if int(r.document_id) % 7 == 0 else fake_topics(*r.title.split()[:3]) for r in rec.itertuples()})
    monkeypatch.setattr(embeddings, "embed", lambda texts, input_type: np.stack([fake_vector(t) for t in texts]) if texts else np.zeros((0, embeddings.DIMENSION)))
    return tmp_path


def run(personas, limit, threshold=0.2):
    args = SimpleNamespace(threshold=threshold, scoring="max", input_type="passage", upsert=False)
    all_personas = match.read_csv(match.PERSONAS_PATH)
    news = match.read_csv(match.NEWS_PATH).set_index("news_id")
    match.run_matching(all_personas[all_personas.persona_id.isin(personas)], news, match.load_recovery(limit), args)
    return pd.read_csv(match.MATCHES_PATH, dtype={"document_id": str})


def source_hashes() -> dict[str, str]:
    files = [match.PERSONAS_PATH, match.NEWS_PATH, match.RECOVERY_PATH, *sorted(match.GRAPHRAG_INPUT.glob("*.txt"))]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def test_pipeline_leaves_source_datasets_unchanged(offline):
    before = source_hashes()
    run(["P01"], 30)
    assert source_hashes() == before


def test_matching_is_reproducible_and_complete(offline):
    first = run(["P01"], 30)
    digest = hashlib.sha256(match.MATCHES_PATH.read_bytes()).hexdigest()
    run(["P01"], 30)
    assert hashlib.sha256(match.MATCHES_PATH.read_bytes()).hexdigest() == digest
    assert len(first) == 30 and not first.duplicated(["persona_id", "document_id"]).any()
    assert (first.included == (first.score >= first.threshold)).all()
    assert first["rank"].tolist() == list(range(1, 31))
    interests = json.loads(match.INTERESTS_PATH.read_text())["P01"]["interests"]
    assert interests and all(i["topic_id"] in TOPIC_IDS and i["label"] for i in interests)
    unlabeled = first[first.document_id.astype(int) % 7 == 0]
    assert len(unlabeled) and (unlabeled.score == match.NO_MATCH_SCORE).all() and not unlabeled.included.any()
    labeled = first[first.best_article_topic_id.notna()]
    assert labeled.best_article_topic_id.isin(TOPIC_IDS).all()


def test_persona_without_usable_interests_still_gets_rows(offline, monkeypatch):
    monkeypatch.setattr(match, "label_mind", lambda news, ids: {i: [] for i in ids})
    matches = run(["P01"], 20)
    assert len(matches) == 20 and (matches.score == match.NO_MATCH_SCORE).all() and not matches.included.any()
    assert json.loads(match.INTERESTS_PATH.read_text())["P01"]["interests"] == []


def test_judgments_refuse_pilot_and_sample_exact_count(offline):
    run(["P01"], 50)
    with pytest.raises(SystemExit, match="run full matching"):
        match.sample_judgments(120)

    run(list(match.read_csv(match.PERSONAS_PATH).persona_id), None)
    match.sample_judgments(120)
    judged = pd.read_csv(match.JUDGMENTS_PATH)
    assert len(judged) == 120 and (judged.persona_id.value_counts() == 12).all()
    assert set(judged.split) == {"calibration", "holdout"} and judged.lede.str.len().gt(0).all()
    with pytest.raises(SystemExit):
        match.evaluate_judgments()  # nothing labeled yet -> insufficient examples

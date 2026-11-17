"""Topic assignments for MIND and ReCOVery articles, chosen from the shared topic dictionary and cached as JSON.

The vocabulary (topic_data/topic_dictionary.json, IPTC Media Topics) is keyed by topic ID. Assignments are kept
separately, keyed by article ID, each a list of 0..MAX_LABELS topic IDs. Every prompt offers the fixed backbone
(IPTC levels 1-2) plus a per-article BM25 shortlist of deeper topics; the model may only return offered IDs.
"""

from __future__ import annotations
import functools
import hashlib
import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

MODEL = "gpt-4.1-mini"
PROMPT_VERSION = "v2-topic-ids"
SEED = 42
MAX_LABELS = 5
LEDE_WORDS = 400
WORKERS = 8
CHECKPOINT_EVERY = 25

ROOT = Path(__file__).resolve().parents[2]
DICTIONARY_PATH = ROOT / "topic_data" / "topic_dictionary.json"
SOURCE_PATH = ROOT / "topic_data" / "topic_dictionary_source.json"

BACKBONE_DEPTH = 2  # IPTC levels 1-2 (137 topics) are offered in every prompt
SHORTLIST_K = 40  # deeper topics added per article by BM25
BM25_K1, BM25_B = 1.2, 0.75
TITLE_WEIGHT = 2.0  # headline terms count double, so a long lede does not drown the subject
SHORTLIST_VERSION = f"bm25-k{SHORTLIST_K}-depth{BACKBONE_DEPTH}-title{TITLE_WEIGHT:g}-ancestors"

SYSTEM_PROMPT = """You assign news articles to topics from a fixed topic dictionary (IPTC Media Topics).

Rules:
- Choose only topic IDs that appear in the candidate lists: the general topics below, or the specific candidates sent with the article. Never invent IDs.
- Return 0 to 5 distinct topic IDs, most central first.
- Prefer the most specific topic that fits; add a broader topic only when the article is genuinely about the broader subject.
- If the article is about COVID-19, include the specific angle it covers (e.g. unemployment benefits, school closures, vaccine, clinical trial), with or without "epidemic and pandemic".
- Topics describe subject matter only. Never express or infer stance, ideology, sentiment, bias, or reliability.
- If no candidate topic fits, return an empty list.

General topics (always available):
{backbone}"""

SCHEMA = {
    "name": "topic_assignment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"topic_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["topic_ids"],
        "additionalProperties": False,
    },
}

STOPWORDS = frozenset(
    "a about after all also an and any are as at be been but by can could did do does for from had has have he her "
    "his how i if in into is it its may more most new not of on one or other our out over said she so some such than "
    "that the their them then there these they this those to up was we were what when which who will with would you "
    "your news".split()
)


def content_sha256(topics: dict) -> str:
    """Same canonical hash as build_topic_dictionary.py: formatting changes are not vocabulary changes."""
    canonical = json.dumps(topics, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Vocabulary:
    topics: dict[str, dict]
    source: str = "IPTC Media Topics"
    version: str = "unversioned"
    sha256: str = field(init=False)

    def __post_init__(self):
        for topic_id, topic in self.topics.items():
            if not topic.get("name") or not topic.get("definition"):
                raise ValueError(f"topic {topic_id} lacks a name or definition")
            if topic.get("parent") and topic["parent"] not in self.topics:
                raise ValueError(f"topic {topic_id} has unknown parent {topic['parent']}")
        object.__setattr__(self, "sha256", content_sha256(self.topics))

    def name(self, topic_id: str) -> str:
        return self.topics[topic_id]["name"]

    def depth(self, topic_id: str) -> int:
        return len(self.ancestors(topic_id)) + 1

    def ancestors(self, topic_id: str) -> list[str]:
        out, node = [], self.topics[topic_id]
        while node.get("parent"):
            out.append(node["parent"])
            node = self.topics[node["parent"]]
        return out

    def header(self) -> dict:
        return {"source": self.source, "version": self.version, "sha256": self.sha256}


def load_dictionary(path: Path = DICTIONARY_PATH, source_path: Path = SOURCE_PATH) -> Vocabulary:
    topics = json.loads(path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    vocab = Vocabulary(topics, source=source["source"], version=source["version"])
    if vocab.sha256 != source["dictionary_sha256"]:
        raise SystemExit(f"{path.name} does not match {source_path.name}; rebuild with build_topic_dictionary.py")
    return vocab


@functools.cache
def vocabulary() -> Vocabulary:
    return load_dictionary()


def stem(word: str) -> str:
    """Plural folding to a shared search token: movie/movies -> movi, company/companies -> compani, car/cars -> car.

    Not a dictionary form: a final -y and -ie both become -i so the singular and the -ies plural collide.
    """
    if len(word) >= 4 and word.endswith("ies"):
        word = word[:-3] + "i"
    elif len(word) >= 4 and word.endswith("ie"):
        word = word[:-1]
    elif len(word) > 4 and word.endswith(("sses", "xes", "ches", "shes")):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if len(word) > 3 and word.endswith("y"):
        word = word[:-1] + "i"
    return word


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.casefold().replace("covid-19", "covid19"))
    return [stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


class Shortlist:
    """Backbone (depth <= BACKBONE_DEPTH) plus BM25 top-k deeper topics and their ancestors; deterministic."""

    def __init__(self, vocab: Vocabulary, k: int = SHORTLIST_K):
        self.vocab, self.k = vocab, k
        self.backbone = sorted(t for t in vocab.topics if vocab.depth(t) <= BACKBONE_DEPTH)
        self.backbone_set = frozenset(self.backbone)
        self.deep = sorted(t for t in vocab.topics if vocab.depth(t) > BACKBONE_DEPTH)
        self.docs = []
        for topic_id in self.deep:
            topic = vocab.topics[topic_id]
            label = " ".join([topic["name"], *topic.get("aliases", [])])
            self.docs.append(Counter(tokens(label) * 2 + tokens(topic["definition"])))  # name/aliases weigh double
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.docs), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - c + 0.5) / (c + 0.5)) for term, c in df.items()}
        self.postings: dict[str, list[int]] = {}
        for j, d in enumerate(self.docs):
            for term in d:
                self.postings.setdefault(term, []).append(j)

    def ranked(self, text: str) -> list[tuple[str, float]]:
        """Deeper topics with a positive BM25 score, best first (ties broken by topic ID)."""
        scores: dict[int, float] = {}
        title = set(tokens(text.split("\n", 1)[0])) if text.startswith("Title:") else set()
        for term in set(tokens(text)):
            weight = TITLE_WEIGHT if term in title else 1.0
            for j in self.postings.get(term, ()):
                tf = self.docs[j][term]
                norm = tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[j] / self.avg_len))
                scores[j] = scores.get(j, 0.0) + weight * self.idf[term] * norm
        return sorted(((self.deep[j], s) for j, s in scores.items()), key=lambda p: (-p[1], p[0]))

    def candidates(self, text: str) -> list[str]:
        """Per-article candidates: BM25 hits in rank order, then their ancestors not already in the backbone."""
        hits = [t for t, _ in self.ranked(text)[: self.k]]
        extra = [a for t in hits for a in self.vocab.ancestors(t) if a not in self.backbone_set]
        return list(dict.fromkeys(hits + extra))

    def offered(self, text: str) -> list[str]:
        """Every ID the model may return for this text: the backbone, then this article's candidates."""
        return self.backbone + self.candidates(text)


def topic_line(vocab: Vocabulary, topic_id: str) -> str:
    return f"{topic_id} | {vocab.name(topic_id)} - {vocab.topics[topic_id]['definition']}"


def system_prompt(shortlist: Shortlist) -> str:
    return SYSTEM_PROMPT.format(backbone="\n".join(topic_line(shortlist.vocab, t) for t in shortlist.backbone))


def user_prompt(shortlist: Shortlist, text: str) -> str:
    lines = "\n".join(topic_line(shortlist.vocab, t) for t in shortlist.candidates(text)) or "(none)"
    return f"Specific candidate topics:\n{lines}\n\nArticle:\n{text}"


class InvalidResponse(ValueError):
    """The model returned IDs that were not offered, or too many."""


def validate(raw_ids, offered: list[str]) -> list[str]:
    """Deduplicated (order kept) IDs; rejects anything not offered and more than MAX_LABELS. [] is valid."""
    ids = list(dict.fromkeys(str(i).strip() for i in raw_ids))
    allowed = set(offered)
    if bad := [i for i in ids if i not in allowed]:
        raise InvalidResponse(f"returned topic IDs that were not offered: {bad}")
    if len(ids) > MAX_LABELS:
        raise InvalidResponse(f"returned {len(ids)} topic IDs, max is {MAX_LABELS}")
    return ids


def mind_text(row: dict) -> str:
    return f"Title: {row['title']}\nAbstract: {row['abstract']}\nSection: {row['category']} / {row['subcategory']}"


def recovery_text(row: dict) -> str:
    lede = " ".join(row["body_text"].split()[:LEDE_WORDS])
    return f"Title: {row['title']}\n\n{lede}"


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


class LabelCache:
    """Article assignments {article_id: [topic_id]} plus source-text hashes, tied to one model/prompt/dictionary.

    A cache built with another model, prompt, shortlist or dictionary (including old free-form caches, which have
    no dictionary header) is refused rather than reused; a changed source text is relabeled.
    """

    def __init__(self, path: Path, vocab: Vocabulary | None = None):
        self.path = path
        self.vocab = vocab or vocabulary()
        self.assignments: dict[str, list[str]] = {}
        self.source_sha: dict[str, str] = {}
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if (stored.get("model"), stored.get("prompt_version"), stored.get("shortlist"), stored.get("dictionary")) \
                    != (MODEL, PROMPT_VERSION, SHORTLIST_VERSION, self.vocab.header()):
                raise SystemExit(
                    f"{path} was built with {stored.get('model')}/{stored.get('prompt_version')}/"
                    f"{stored.get('shortlist')} and dictionary {stored.get('dictionary')}; current is "
                    f"{MODEL}/{PROMPT_VERSION}/{SHORTLIST_VERSION} and {self.vocab.header()}. Delete it to relabel."
                )
            unknown = sorted({t for ids in stored["assignments"].values() for t in ids} - set(self.vocab.topics))
            if unknown:
                raise SystemExit(f"{path} assigns topic IDs missing from the dictionary: {unknown[:5]}")
            self.assignments, self.source_sha = stored["assignments"], stored["source_sha"]

    def fresh(self, article_id: str, text: str) -> bool:
        return article_id in self.assignments and self.source_sha.get(article_id) == digest(text)

    def put(self, article_id: str, text: str, topic_ids: list[str]) -> None:
        self.assignments[article_id] = topic_ids
        self.source_sha[article_id] = digest(text)

    def save(self) -> None:
        order = sorted(self.assignments, key=lambda i: (len(i), i))
        payload = {
            "model": MODEL, "prompt_version": PROMPT_VERSION, "shortlist": SHORTLIST_VERSION,
            "dictionary": self.vocab.header(),
            "assignments": {i: self.assignments[i] for i in order},
            "source_sha": {i: self.source_sha[i] for i in order},
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)


def request_topics(client, system: str, user: str) -> list:
    """One model call; returns the raw `topic_ids` list (validated by the caller)."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        seed=SEED,
        response_format={"type": "json_schema", "json_schema": SCHEMA},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return json.loads(response.choices[0].message.content)["topic_ids"]


def call_llm(client, text: str, shortlist: Shortlist) -> list[str]:
    return validate(request_topics(client, system_prompt(shortlist), user_prompt(shortlist, text)), shortlist.offered(text))


def make_client():
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or os.environ["GRAPHRAG_API_KEY"], max_retries=5)


def label_articles(cache: LabelCache, texts: dict[str, str]) -> dict[str, list[str]]:
    """Assign topics to every id in `texts` that is missing or stale; progress is checkpointed and kept on failure."""
    todo = [i for i, text in texts.items() if not cache.fresh(i, text)]
    failed: dict[str, str] = {}
    if todo:
        client, shortlist = make_client(), Shortlist(cache.vocab)
        print(f"labeling {len(todo)} articles -> {cache.path.name}")
        try:
            with ThreadPoolExecutor(WORKERS) as pool:
                futures = {pool.submit(call_llm, client, texts[i], shortlist): i for i in todo}
                for n, future in enumerate(as_completed(futures), 1):
                    i = futures[future]
                    try:
                        cache.put(i, texts[i], future.result())  # [] is a valid, cached result
                    except Exception as error:  # keep going; the cache retains everything that succeeded
                        failed[i] = repr(error)
                    if n % CHECKPOINT_EVERY == 0:
                        cache.save()
        finally:
            cache.save()
    if failed:
        raise RuntimeError(f"{len(failed)} articles failed (rerun to retry): {dict(list(failed.items())[:5])}")
    return {i: cache.assignments[i] for i in texts}

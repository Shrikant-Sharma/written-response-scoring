"""The features of one written response. 20 hand-made ones in three groups, each traceable to something a rater could see.

Nothing here looks at a score. Two things are learned from the training responses only, in fit():
  - the 3,000 most common words, so that "share of uncommon words" has a fixed meaning;
  - per prompt, what the top quarter and the bottom quarter of responses talk about (two TF-IDF centroids).
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

WORD = re.compile(r"[A-Za-z']+")
SENTENCE_END = re.compile(r"[.!?]+")
MASK = re.compile(r"@[A-Z]+\d*")                         # the publisher replaced names, places and dates with tags like @PERSON1
LINKING = {"because", "however", "therefore", "although", "since", "while", "but", "so", "also", "furthermore", "moreover", "instead", "thus",
           "unless", "whereas", "first", "second", "finally", "then", "though"}
GROUPS = {
    "structure": ["words", "log_words", "sentences", "words_per_sentence", "longest_sentence", "commas_per_100", "questions_and_exclamations_per_100"],
    "language": ["distinct_words", "distinct_share", "letters_per_word", "long_word_share", "uncommon_word_share", "linking_words_per_100",
                 "capital_starts_share", "masked_names_per_100", "quote_marks_per_100", "digits_per_100"],
    "content": ["like_top_responses", "like_bottom_responses", "top_minus_bottom"],
}
FEATURES = [f for names in GROUPS.values() for f in names]
MEANING = {
    "words": "number of words", "log_words": "number of words, on a log scale", "sentences": "number of sentences",
    "words_per_sentence": "words per sentence", "longest_sentence": "words in the longest sentence",
    "commas_per_100": "commas per 100 words", "questions_and_exclamations_per_100": "question and exclamation marks per 100 words",
    "distinct_words": "number of different words", "distinct_share": "different words per square root of words",
    "letters_per_word": "letters per word", "long_word_share": "share of words with 7 letters or more",
    "uncommon_word_share": "share of words outside the 3,000 most common (rare words and misspellings)",
    "linking_words_per_100": "linking words (because, however, ...) per 100 words",
    "capital_starts_share": "share of sentences that start with a capital letter",
    "masked_names_per_100": "names, places and dates (masked by the publisher) per 100 words",
    "quote_marks_per_100": "quotation marks per 100 words", "digits_per_100": "digits per 100 words",
    "like_top_responses": "similarity of its wording to the top quarter of training responses on the same prompt",
    "like_bottom_responses": "similarity of its wording to the bottom quarter of training responses on the same prompt",
    "top_minus_bottom": "how much more its wording resembles the top quarter than the bottom quarter",
}


@lru_cache(maxsize=None)
def surface(text):
    """The 17 features that need nothing but the text itself (uncommon_word_share is filled in later)."""
    words = [w.lower() for w in WORD.findall(MASK.sub(" ", text))]
    n = max(len(words), 1)
    pieces = [p for p in SENTENCE_END.split(text) if p.strip()]
    lengths = [len(WORD.findall(MASK.sub(" ", p))) for p in pieces] or [0]
    starts = [p.strip()[0] for p in pieces] or ["a"]
    per_100 = 100.0 / n
    return {
        "words": len(words), "log_words": float(np.log1p(len(words))), "sentences": len(pieces),
        "words_per_sentence": len(words) / max(len(pieces), 1), "longest_sentence": max(lengths),
        "commas_per_100": text.count(",") * per_100, "questions_and_exclamations_per_100": (text.count("?") + text.count("!")) * per_100,
        "distinct_words": len(set(words)), "distinct_share": len(set(words)) / np.sqrt(n),
        "letters_per_word": sum(len(w) for w in words) / n, "long_word_share": sum(len(w) >= 7 for w in words) / n,
        "linking_words_per_100": sum(w in LINKING for w in words) * per_100,
        "capital_starts_share": sum(c.isupper() or c == "@" for c in starts) / len(starts),
        "masked_names_per_100": len(MASK.findall(text)) * per_100, "quote_marks_per_100": text.count('"') * per_100,
        "digits_per_100": sum(c.isdigit() for c in MASK.sub(" ", text)) * per_100,
    }, words


class Features:
    def fit(self, text, prompt, score):
        """Learn the common words and the two centroids per prompt. Call this with training responses only."""
        text, prompt, score = list(text), np.asarray(prompt), np.asarray(score, dtype=float)
        counts = {}
        for t in text:
            for w in surface(t)[1]:
                counts[w] = counts.get(w, 0) + 1
        self.common = set(sorted(counts, key=lambda w: (-counts[w], w))[:3000])
        self.tfidf = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=3, max_features=20000, token_pattern=r"[A-Za-z']{2,}")
        M = self.tfidf.fit_transform(text)
        self.top, self.bottom = {}, {}
        for p in np.unique(prompt):
            rows = np.flatnonzero(prompt == p)
            order = rows[np.argsort(score[rows], kind="stable")]
            quarter = max(len(rows) // 4, 1)
            self.bottom[int(p)] = unit(np.asarray(M[order[:quarter]].mean(axis=0)).ravel())
            self.top[int(p)] = unit(np.asarray(M[order[-quarter:]].mean(axis=0)).ravel())
        return self

    def transform(self, text, prompt):
        text, prompt = list(text), np.asarray(prompt)
        rows = []
        for t in text:
            f, words = surface(t)
            f = dict(f)
            f["uncommon_word_share"] = sum(w not in self.common for w in words) / max(len(words), 1)
            rows.append(f)
        out = pd.DataFrame(rows)
        M = self.tfidf.transform(text)                                 # rows have length 1, so a dot product is a cosine
        out["like_top_responses"] = [float(M[i].dot(self.top[int(p)])[0]) for i, p in enumerate(prompt)]
        out["like_bottom_responses"] = [float(M[i].dot(self.bottom[int(p)])[0]) for i, p in enumerate(prompt)]
        out["top_minus_bottom"] = out["like_top_responses"] - out["like_bottom_responses"]
        return out[FEATURES]


def fit_with_out_of_fold(text, prompt, score, fold):
    """Fit on all the responses given, and also return their own features computed out of fold.
    The content features compare a response with the top and bottom responses. For a training response those
    groups must not include the response itself, or the feature would carry its own score."""
    text, prompt, score, fold = np.asarray(list(text), dtype=object), np.asarray(prompt), np.asarray(score, dtype=float), np.asarray(fold)
    own = pd.DataFrame(index=range(len(text)), columns=FEATURES, dtype=float)
    for k in np.unique(fold):
        rest, held = fold != k, fold == k
        own.loc[held] = Features().fit(text[rest], prompt[rest], score[rest]).transform(text[held], prompt[held]).to_numpy()
    return Features().fit(text, prompt, score), own


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


class WithinPrompt:
    """Put each feature on the scale of its own prompt: how far a response sits from that prompt's training average, in standard deviations.
    A 300-word answer is long for one prompt and short for another, so raw values cannot be compared across prompts."""

    def fit(self, X, prompt):
        prompt = np.asarray(prompt)
        self.mean = {int(p): X[prompt == p].mean() for p in np.unique(prompt)}
        self.sd = {int(p): X[prompt == p].std(ddof=0).replace(0, 1.0) for p in np.unique(prompt)}
        self.all_mean, self.all_sd = X.mean(), X.std(ddof=0).replace(0, 1.0)
        return self

    def transform(self, X, prompt, unseen_prompt=False):
        """unseen_prompt=True is for a prompt that was not in training: its own responses set the scale (no scores are used)."""
        prompt = np.asarray(prompt)
        Z = X.copy().astype(float)
        for p in np.unique(prompt):
            rows = prompt == p
            if unseen_prompt or int(p) not in self.mean:
                Z.loc[rows] = (X.loc[rows] - X.loc[rows].mean()) / X.loc[rows].std(ddof=0).replace(0, 1.0)
            else:
                Z.loc[rows] = (X.loc[rows] - self.mean[int(p)]) / self.sd[int(p)]
        return Z.clip(-6, 6)

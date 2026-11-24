"""Checks that need no data file.

    python tests/test_all.py
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import FEATURES, Features, fit_with_out_of_fold, surface
from scoring import Scorer, measures, to_final, weighted_kappa

FAILED = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        FAILED.append(name)


def made_up(n=240, seed=3):
    """Responses whose quality is known: better ones are longer, use rarer words and stay on their prompt's topic."""
    rng = np.random.default_rng(seed)
    topic = {1: "computers screens typing internet email games".split(), 2: "library books censorship shelves reading freedom".split()}
    plain = "the a and it is was to of in that this with for on at by".split()
    rows = []
    for i in range(n):
        prompt, quality = 1 + i % 2, rng.random()
        words = list(rng.choice(plain, size=int(20 + 80 * quality))) + list(rng.choice(topic[prompt], size=int(2 + 20 * quality)))
        rng.shuffle(words)
        text = " ".join(words[:len(words) // 2]) + ". " + " ".join(words[len(words) // 2:]) + "."
        rows.append({"response_id": i, "prompt": prompt, "text": text.capitalize(), "score": quality, "fold": i % 5, "final_score": int(round(1 + quality * 5)) if prompt == 2 else int(round(2 + quality * 10))})
    return pd.DataFrame(rows)


def main():
    f, words = surface('Dear @PERSON1, I think computers help. However, do they? "Yes," I said 3 times!')
    check("surface: counts words, sentences and marks of one known sentence",
          f["words"] == 12 and f["sentences"] == 3 and f["longest_sentence"] == 5 and abs(f["masked_names_per_100"] - 100 / 12) < 1e-9
          and abs(f["quote_marks_per_100"] - 200 / 12) < 1e-9 and abs(f["digits_per_100"] - 100 / 12) < 1e-9 and abs(f["linking_words_per_100"] - 100 / 12) < 1e-9)
    check("surface: a masked name is not counted as a word", "person" not in words)

    data = made_up()
    fitted = Features().fit(data.text, data.prompt, data.score)
    X = fitted.transform(data.text, data.prompt)
    check("features: 20 columns in the documented order, no missing values", list(X.columns) == FEATURES and len(FEATURES) == 20 and not X.isna().any().any())
    check("features: transform never needs a score", X.equals(fitted.transform(data.text, data.prompt)))

    _, own = fit_with_out_of_fold(data.text, data.prompt, data.score, data.fold)
    changed = data.score.copy(); changed.iloc[0] = 1 - changed.iloc[0]
    _, own_changed = fit_with_out_of_fold(data.text, data.prompt, changed, data.fold)
    check("out of fold: changing a response's own score does not change its own content features", np.allclose(own.iloc[0], own_changed.iloc[0]))
    inside = fitted.transform(data.text.iloc[:1], data.prompt.iloc[:1])
    inside_changed = Features().fit(data.text, data.prompt, changed).transform(data.text.iloc[:1], data.prompt.iloc[:1])
    check("and without out-of-fold it would (the leak this guards against)", not np.allclose(inside, inside_changed))

    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 5, 500), rng.integers(0, 5, 500)
    check("kappa: equals scikit-learn's quadratic weighted kappa", abs(weighted_kappa(a, b, 0, 4) - cohen_kappa_score(a, b, weights="quadratic", labels=list(range(5)))) < 1e-9)
    check("kappa: 1 for identical scores, about 0 for unrelated ones", abs(weighted_kappa(a, a, 0, 4) - 1) < 1e-9 and abs(weighted_kappa(a, b, 0, 4)) < 0.1)
    check("to_final: maps 0, 0.5 and 1 onto each prompt's own scale", list(to_final(np.array([0, 0.5, 1.0, 0.5]), np.array([1, 1, 1, 8]))) == [2, 7, 12, 30])

    scorer = Scorer().fit(data)
    m = measures(data.score, scorer.predict("random forest", data.text, data.prompt), data.prompt, data.final_score)
    base = measures(data.score, scorer.predict("prompt average", data.text, data.prompt), data.prompt, data.final_score)
    check("scorer: on made-up responses with a known pattern, the forest beats the prompt average", m["mae_points"] < base["mae_points"] / 2)

    sql = (Path(__file__).resolve().parents[1] / "sql")
    con = duckdb.connect()
    rows = pd.DataFrame({"essay_id": [str(i) for i in range(1, 2001)] + ["5000"], "essay_set": ["1"] * 1000 + ["2"] * 1000 + ["1"],
                         "essay": [f"text number {i}" for i in range(1, 2001)] + ["TEXT  number 7"], "rater1_domain1": ["3"] * 2001, "rater2_domain1": ["3"] * 2001,
                         "rater3_domain1": [None] * 2001, "domain1_score": ["6"] * 1000 + ["3"] * 1000 + ["8"]})
    con.register("file_rows", rows)
    for name in ("01_load.sql", "02_duplicates.sql", "03_score_and_split.sql"):
        con.execute((sql / name).read_text(encoding="utf-8"))
    by_group = con.execute("SELECT coalesce(sum(copies - 1), 0) FROM duplicate_groups").fetchone()[0]
    by_window = con.execute("SELECT count(*) FROM numbered WHERE copy_number > 1").fetchone()[0]
    check("sql: both ways find the one duplicate, which differs only in case and spacing", by_group == 1 and by_window == 1)
    out = con.execute("SELECT * FROM prepared").df()
    check("sql: the first copy is kept and every response lands in exactly one part", len(out) == 2000 and 5000 not in set(out.response_id) and set(out.part) == {"train", "tune", "test"})
    again = duckdb.connect(); again.register("file_rows", rows.sample(frac=1, random_state=1))
    for name in ("01_load.sql", "02_duplicates.sql", "03_score_and_split.sql"):
        again.execute((sql / name).read_text(encoding="utf-8"))
    out2 = again.execute("SELECT * FROM prepared").df()
    check("sql: the split and the folds do not depend on the order of the rows", out[["response_id", "part", "fold"]].equals(out2[["response_id", "part", "fold"]]))
    share = out.part.value_counts(normalize=True)
    check("sql: the parts are close to 60, 20 and 20 percent", abs(share["train"] - 0.6) < 0.04 and abs(share["tune"] - 0.2) < 0.04)
    check("sql: scores are rescaled within the prompt", abs(out[out.prompt == 1].score.iloc[0] - 0.4) < 1e-9 and abs(out[out.prompt == 2].score.iloc[0] - 0.4) < 1e-9)

    print("\nAll checks passed." if not FAILED else f"\n{len(FAILED)} check(s) failed.")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()

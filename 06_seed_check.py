"""Extension, part 1. The seed check: is the forest's gain over ridge regression steady when the forest is grown again?

    python 06_seed_check.py            about 2 minutes on 2 cores

The interval in step 3 redraws the responses but holds the two trained models fixed. A random forest is grown with
random draws, so a different seed gives a slightly different forest. Here the forest is grown from five seeds on the
same training part, and its gain over ridge regression is quoted as a range. Ridge regression and the boosted trees
(as set in scoring.py) involve no random draw, so they are fitted once.

The first seed is the project's own. Its row must repeat step 3: that is the check that this file rebuilds the same model.

Nothing that exists is changed. Writes results/seed_check.json and results/seed_check.md.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from common import RESULTS, SEED, WORK, read_json, write_json
from features import WithinPrompt, fit_with_out_of_fold
from scoring import Scorer, measures, one_hot, paired_interval

SEEDS = [SEED, 1, 2, 3, 4]


def fitted_design(train):
    """The same features, scaling and design table that Scorer.fit builds, kept so that a model can be fitted on them directly."""
    text, prompt, score = train["text"].to_numpy(), train["prompt"].to_numpy(), train["score"].to_numpy()
    scorer = Scorer()
    scorer.features, own = fit_with_out_of_fold(text, prompt, score, train["fold"].to_numpy())
    scorer.scale = WithinPrompt().fit(own, prompt)
    design = pd.concat([scorer.scale.transform(own, prompt).reset_index(drop=True), one_hot(prompt)], axis=1)
    return scorer, design, score


def main():
    t0 = time.time()
    data = pd.read_parquet(WORK / "prepared.parquet")
    train, tune, test = (data[data["part"] == part].reset_index(drop=True) for part in ("train", "tune", "test"))
    scorer, design, score = fitted_design(train)
    parts = {"tune": (tune, scorer.design(tune["text"], tune["prompt"])), "test": (test, scorer.design(test["text"], test["prompt"]))}
    predict = lambda model, X: np.clip(model.predict(X), 0, 1)
    error = lambda frame, predicted: measures(frame["score"], predicted, frame["prompt"], frame["final_score"])["mae_points"]

    ridge = Ridge(alpha=10.0).fit(design, score)
    boosted = HistGradientBoostingRegressor(learning_rate=0.06, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                                            early_stopping=False, random_state=SEED).fit(design, score)
    by_ridge = {name: predict(ridge, X) for name, (_, X) in parts.items()}
    by_boosted = {name: predict(boosted, X) for name, (_, X) in parts.items()}
    rows = []
    for seed in SEEDS:
        forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=2, random_state=seed).fit(design, score)
        row = {"seed": seed}
        for name, (frame, X) in parts.items():
            by_forest = predict(forest, X)
            row[name] = {"forest_error": error(frame, by_forest), "over_ridge": paired_interval(frame["score"], by_ridge[name], by_forest),
                         "over_boosted": paired_interval(frame["score"], by_boosted[name], by_forest)}
        rows.append(row)
        print(f"seed {seed} done ({time.time() - t0:.0f} s)", flush=True)

    step_3 = read_json(RESULTS / "models.json")
    repeats = abs(rows[0]["tune"]["forest_error"] - step_3["tune"]["random forest"]["mae_points"]) < 1e-6 and abs(rows[0]["test"]["forest_error"] - step_3["test"]["random forest"]["mae_points"]) < 1e-6
    spread = lambda name, key: [min(r[name][key]["gain_points"] for r in rows), max(r[name][key]["gain_points"] for r in rows)]
    result = {"seeds": SEEDS, "first_seed_repeats_step_3": bool(repeats),
              "ridge_error": {name: error(frame, by_ridge[name]) for name, (frame, _) in parts.items()},
              "boosted_error": {name: error(frame, by_boosted[name]) for name, (frame, _) in parts.items()},
              "by_seed": rows,
              "gain_over_ridge_range": {name: spread(name, "over_ridge") for name in parts},
              "lowest_interval_end_over_ridge": {name: min(r[name]["over_ridge"]["low"] for r in rows) for name in parts},
              "gain_over_boosted_range": {name: spread(name, "over_boosted") for name in parts},
              "seeds_where_forest_beats_boosted_clearly": {name: sum(r[name]["over_boosted"]["low"] > 0 for r in rows) for name in parts},
              "seconds": round(time.time() - t0)}
    write_json(RESULTS / "seed_check.json", result)

    L = ["# Extension, part 1: the seed check", "",
         f"The forest is grown from {len(SEEDS)} seeds on the same training part. Ridge regression and the boosted trees involve no random draw and are fitted once. Errors and gains are in points of 100.", "",
         f"The first seed is the project's own. Its row {'repeats' if repeats else 'DOES NOT repeat'} step 3.", ""]
    for name in parts:
        L += [f"## The {name} part", "", f"Ridge regression: {result['ridge_error'][name]:.2f}. Boosted trees: {result['boosted_error'][name]:.2f}.", "",
              "| Seed | Forest, mean error | Gain over ridge | 95% interval | Gain over boosted trees | 95% interval |", "|---|---|---|---|---|---|"]
        L += [f"| {r['seed']} | {r[name]['forest_error']:.2f} | {r[name]['over_ridge']['gain_points']:+.2f} | {r[name]['over_ridge']['low']:+.2f} to {r[name]['over_ridge']['high']:+.2f} | "
              f"{r[name]['over_boosted']['gain_points']:+.2f} | {r[name]['over_boosted']['low']:+.2f} to {r[name]['over_boosted']['high']:+.2f} |" for r in rows]
        a, b = result["gain_over_ridge_range"][name], result["gain_over_boosted_range"][name]
        L += ["", f"Over ridge regression the gain runs from {a[0]:.2f} to {a[1]:.2f} points across the seeds, and the lowest end of any interval is {result['lowest_interval_end_over_ridge'][name]:+.2f}. "
              f"Over the boosted trees it runs from {b[0]:+.2f} to {b[1]:+.2f}, and the interval is clear of zero for {result['seeds_where_forest_beats_boosted_clearly'][name]} of {len(SEEDS)} seeds.", ""]
    L += ["What this does not cover: the training responses themselves are the same in every row. Only the forest's own random draws change.", ""]
    (RESULTS / "seed_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s. See results/seed_check.md")


if __name__ == "__main__":
    main()

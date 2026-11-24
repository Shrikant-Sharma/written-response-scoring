"""Step 3. Compare six ways of predicting the score, choose one, and score all six on the test part.

    python 03_models.py            about 1.5 minutes on 2 cores

The target is the final human score of a response, put on a 0 to 1 scale within its prompt. It is a number, so this is regression.
  - Two baselines: the prompt's average score, and the length of the response alone.
  - Words only: a ridge regression on the words themselves, one per prompt.
  - Three models on the 20 hand-made features: ridge regression, a random forest, boosted trees.

How the choice is made:
  1. Five-fold cross-validation on the training part ranks the six. Everything that is learned from data, the
     features included, is learned again inside each fold.
  2. The tuning part settles one question with an interval: does the best model beat the plain ridge regression
     by more than chance? If the interval for the gain includes zero, the ridge regression is kept.
  3. All six are then scored on the test part. The choice is made in steps 1 and 2 and does not look at it.

The settings of each method are written by hand in scoring.py. No search over settings is run.

Writes results/models.json, results/models.md and data/work/predictions.parquet.
"""
import time

import numpy as np
import pandas as pd

from common import RESULTS, SCALE, WORK, write_json
from scoring import MODELS, USES_FEATURES, Scorer, measures, paired_interval, to_final


def main():
    t0 = time.time()
    data = pd.read_parquet(WORK / "prepared.parquet")
    train, tune, test = (data[data["part"] == part].reset_index(drop=True) for part in ("train", "tune", "test"))

    # 1. cross-validation on the training part
    held = {kind: np.zeros(len(train)) for kind in MODELS}
    by_fold = {kind: [] for kind in MODELS}
    for k in sorted(train["fold"].unique()):
        inside, outside = train[train["fold"] != k].reset_index(drop=True), train["fold"] == k
        scorer = Scorer().fit(inside)
        for kind in MODELS:
            held[kind][outside.to_numpy()] = scorer.predict(kind, train.loc[outside, "text"], train.loc[outside, "prompt"])
            by_fold[kind].append(measures(train.loc[outside, "score"], held[kind][outside.to_numpy()], train.loc[outside, "prompt"], train.loc[outside, "final_score"])["mae_points"])
        print(f"fold {k} done ({time.time() - t0:.0f} s)", flush=True)
    cv = {kind: {**measures(train["score"], held[kind], train["prompt"], train["final_score"]), "mae_by_fold": by_fold[kind],
                 "mae_fold_sd": float(np.std(by_fold[kind], ddof=1))} for kind in MODELS}
    best = min(USES_FEATURES, key=lambda kind: cv[kind]["mae_points"])

    # 2. the tuning part: is the best model better than the plain ridge regression by more than chance?
    scorer = Scorer().fit(train)
    on_tune = {kind: scorer.predict(kind, tune["text"], tune["prompt"]) for kind in MODELS}
    tune_measures = {kind: measures(tune["score"], on_tune[kind], tune["prompt"], tune["final_score"]) for kind in MODELS}
    against_ridge = paired_interval(tune["score"], on_tune["ridge regression"], on_tune[best]) if best != "ridge regression" else None
    chosen = best if against_ridge is None or against_ridge["low"] > 0 else "ridge regression"
    steps = {"length only over prompt average": paired_interval(tune["score"], on_tune["prompt average"], on_tune["length only"]),
             "ridge regression over length only": paired_interval(tune["score"], on_tune["length only"], on_tune["ridge regression"]),
             f"{best} over ridge regression": against_ridge,
             f"{chosen} over words only": paired_interval(tune["score"], on_tune["words only"], on_tune[chosen])}

    # 3. the test part. The choice above does not depend on anything below.
    on_test = {kind: scorer.predict(kind, test["text"], test["prompt"]) for kind in MODELS}
    test_measures = {kind: measures(test["score"], on_test[kind], test["prompt"], test["final_score"]) for kind in MODELS}
    over_length = paired_interval(test["score"], on_test["length only"], on_test[chosen])
    base, final = test_measures["length only"], test_measures[chosen]
    improvement = {"baseline": "length only", "model": chosen,
                   "exact_agreement_baseline": base["exact"], "exact_agreement_model": final["exact"],
                   "exact_agreement_gain_in_points": (final["exact"] - base["exact"]) * 100, "exact_agreement_gain_relative": final["exact"] / base["exact"] - 1,
                   "mae_baseline": base["mae_points"], "mae_model": final["mae_points"], "mae_cut_relative": 1 - final["mae_points"] / base["mae_points"],
                   "mae_gain_interval": over_length}
    importance = None
    if chosen in ("random forest", "boosted trees"):
        from sklearn.inspection import permutation_importance
        design = scorer.design(tune["text"], tune["prompt"])
        r = permutation_importance(scorer.models[chosen], design, tune["score"], scoring="neg_mean_absolute_error", n_repeats=5, random_state=0, n_jobs=2)
        importance = sorted(({"feature": c, "mae_rise_points": float(m * 100)} for c, m in zip(design.columns, r.importances_mean) if not c.startswith("prompt_")), key=lambda d: -d["mae_rise_points"])

    out = pd.concat([tune.assign(part="tune", **{kind.replace(" ", "_"): on_tune[kind] for kind in MODELS}),
                     test.assign(part="test", **{kind.replace(" ", "_"): on_test[kind] for kind in MODELS})])
    out.drop(columns=["text"]).to_parquet(WORK / "predictions.parquet", index=False)
    result = {"parts": {"train": len(train), "tune": len(tune), "test": len(test)}, "features": 20, "models": MODELS, "cross_validation": cv, "best_in_cross_validation": best,
              "tune": tune_measures, "steps_on_tune": steps, "chosen": chosen, "test": test_measures, "improvement_on_test": improvement, "importance_on_tune": importance,
              "seconds": round(time.time() - t0)}
    write_json(RESULTS / "models.json", result)
    write_report(result)
    print(f"done in {time.time() - t0:.0f} s. Chosen: {chosen}. See results/models.md")


def write_report(r):
    from features import MEANING
    row = lambda kind, m: f"| {kind.capitalize()} | {m['mae_points']:.2f} | {m['r2']:.3f} | {m['exact']:.1%} | {m['within_one']:.1%} | {m['kappa_mean']:.3f} |"
    head = ["| Method | Mean error, points of 100 | R squared | Same final score | Within one point | Agreement (kappa) |", "|---|---|---|---|---|---|"]
    cv, i = r["cross_validation"], r["improvement_on_test"]
    L = ["# Step 3: six ways to predict the score", "",
         f"{r['parts']['train']:,} responses to learn from, {r['parts']['tune']:,} to choose on, {r['parts']['test']:,} for the test. The score is on a 0 to 1 scale within its prompt; errors are shown in points of 100.", "",
         "## Cross-validation on the training part (five folds)", "",
         "| Method | Mean error, points of 100 | Spread over the folds | R squared | Same final score | Agreement (kappa) |", "|---|---|---|---|---|---|"]
    L += [f"| {kind.capitalize()} | {cv[kind]['mae_points']:.2f} | {cv[kind]['mae_fold_sd']:.2f} | {cv[kind]['r2']:.3f} | {cv[kind]['exact']:.1%} | {cv[kind]['kappa_mean']:.3f} |" for kind in r["models"]]
    L += ["", f"Best of the three feature models in cross-validation: {r['best_in_cross_validation']}.", "", "## The tuning part", ""] + head + [row(kind, r["tune"][kind]) for kind in r["models"]]
    L += ["", "Does each step pay for itself? Cut in mean error, in points, with a 95% interval from redrawing responses 2,000 times.", "", "| Step | Cut in mean error | 95% interval |", "|---|---|---|"]
    L += [f"| {name.capitalize()} | {g['gain_points']:+.2f} | {g['low']:+.2f} to {g['high']:+.2f} |" for name, g in r["steps_on_tune"].items() if g]
    L += ["", f"Chosen: {r['chosen']}.", "", "## The test part", ""] + head + [row(kind, r["test"][kind]) for kind in r["models"]]
    L += ["", f"Against the length-only baseline, the chosen model gives the same final score as the human raters on {i['exact_agreement_model']:.1%} of test responses, up from {i['exact_agreement_baseline']:.1%}. "
          f"That is {i['exact_agreement_gain_in_points']:+.1f} percentage points, or {i['exact_agreement_gain_relative']:+.0%} in relative terms: two ways of saying the same result, and they are not interchangeable. "
          f"Mean error falls from {i['mae_baseline']:.2f} to {i['mae_model']:.2f} points ({i['mae_gain_interval']['gain_points']:+.2f}; 95% interval {i['mae_gain_interval']['low']:+.2f} to {i['mae_gain_interval']['high']:+.2f}).", "",
          "Agreement by prompt on the test part (quadratic weighted kappa against the final score):", "", "| Prompt | Score scale | Length only | Chosen model |", "|---|---|---|---|"]
    L += [f"| {p} | {SCALE[int(p)][0]} to {SCALE[int(p)][1]} | {r['test']['length only']['kappa_by_prompt'][p]:.3f} | {r['test'][r['chosen']]['kappa_by_prompt'][p]:.3f} |" for p in r["test"][r["chosen"]]["kappa_by_prompt"]]
    if r["importance_on_tune"]:
        L += ["", "## What the chosen model leans on", "", "Each feature is shuffled in turn on the tuning part. The rise in mean error says how much the model needs it.", "", "| Feature | Rise in mean error, points |", "|---|---|"]
        L += [f"| {MEANING[d['feature']]} | {d['mae_rise_points']:.2f} |" for d in r["importance_on_tune"][:8]]
    (RESULTS / "models.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

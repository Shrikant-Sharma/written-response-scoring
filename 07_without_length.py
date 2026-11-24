"""Extension, part 2. My change: the forest without its length features.

    python 07_without_length.py            about 2 minutes on 2 cores

The question: how much of the model is length, and is it harder to fool without it?

Three forests are grown on the same training part, with the same settings and seed:
  - full            all 20 features (this must repeat step 3)
  - no size counts  without number of words, its log, and number of sentences
  - no length       without those three and without the number of different words, which also grows with length

Each is scored on the tune and test parts. Then the paste-twice probe of step 4 is run on each: the same test
response pasted twice, which a human rater would not reward.

Nothing that exists is changed. Writes results/without_length.json and results/without_length.md.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from common import RESULTS, SEED, WORK, read_json, write_json
from features import WithinPrompt, fit_with_out_of_fold
from scoring import Scorer, measures, one_hot, paired_interval

SIZE_COUNTS = ["words", "log_words", "sentences"]
VARIANTS = {"full": [], "no size counts": SIZE_COUNTS, "no length": SIZE_COUNTS + ["distinct_words"]}


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
    on = {"tune": (tune, scorer.design(tune["text"], tune["prompt"])), "test": (test, scorer.design(test["text"], test["prompt"]))}
    twice = scorer.design(np.array([t + " " + t for t in test["text"]], dtype=object), test["prompt"])
    print(f"features ready ({time.time() - t0:.0f} s)", flush=True)

    predicted, result = {}, {"variants": {}}
    for name, dropped in VARIANTS.items():
        keep = [c for c in design.columns if c not in dropped]
        forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=2, random_state=SEED).fit(design[keep], score)
        predict = lambda X: np.clip(forest.predict(X[keep]), 0, 1)
        predicted[name] = {part: predict(X) for part, (_, X) in on.items()}
        before, after = predicted[name]["test"], predict(twice)
        entry = {"dropped": dropped, "features_left": len(keep) - 8}
        for part, (frame, _) in on.items():
            m = measures(frame["score"], predicted[name][part], frame["prompt"], frame["final_score"])
            entry[part] = {k: m[k] for k in ("mae_points", "r2", "exact", "kappa_mean")}
            if name != "full":
                entry[part]["cost_against_full"] = paired_interval(frame["score"], predicted[name][part], predicted["full"][part])
        entry["pasted_twice"] = {"mean_score_before": float(before.mean() * 100), "mean_score_after": float(after.mean() * 100),
                                 "share_scored_higher": float((after > before + 1e-9).mean()), "share_scored_lower": float((after < before - 1e-9).mean())}
        result["variants"][name] = entry
        print(f"{name} done ({time.time() - t0:.0f} s)", flush=True)

    step_3, step_4 = read_json(RESULTS / "models.json"), read_json(RESULTS / "checks.json")
    full = result["variants"]["full"]
    result["full_repeats_step_3_and_4"] = bool(abs(full["test"]["mae_points"] - step_3["test"]["random forest"]["mae_points"]) < 1e-6
                                               and abs(full["pasted_twice"]["share_scored_higher"] - step_4["probes"]["the same text pasted twice"]["share_scored_higher"]) < 1e-9)
    result["length_only_error"] = {"tune": step_3["tune"]["length only"]["mae_points"], "test": step_3["test"]["length only"]["mae_points"]}
    result["seconds"] = round(time.time() - t0)
    write_json(RESULTS / "without_length.json", result)

    V = result["variants"]
    L = ["# Extension, part 2: the forest without its length features", "",
         "Three forests on the same training part, with the same settings and seed. Errors are in points of 100.", "",
         f"The full forest {'repeats' if result['full_repeats_step_3_and_4'] else 'DOES NOT repeat'} step 3 and the paste-twice probe of step 4.", "",
         "| Forest | Features | Taken out |", "|---|---|---|"]
    L += [f"| {name.capitalize()} | {V[name]['features_left']} | {', '.join(V[name]['dropped']) or 'nothing'} |" for name in V]
    for part in ("tune", "test"):
        L += ["", f"## Accuracy on the {part} part", "", f"Length alone, from step 3: {result['length_only_error'][part]:.2f}.", "",
              "| Forest | Mean error | R squared | Same final score | Agreement (kappa) | Cost against the full forest | 95% interval |", "|---|---|---|---|---|---|---|"]
        for name in V:
            m, c = V[name][part], V[name][part].get("cost_against_full")
            cost = f"{c['gain_points']:+.2f} | {c['low']:+.2f} to {c['high']:+.2f}" if c else " | "
            L.append(f"| {name.capitalize()} | {m['mae_points']:.2f} | {m['r2']:.3f} | {m['exact']:.1%} | {m['kappa_mean']:.3f} | {cost} |")
    L += ["", "The cost is how many points of mean error the full forest saves over that forest, with a 95% interval from redrawing responses 2,000 times.", "",
          "## The same text pasted twice, on the test part", "", "| Forest | Mean predicted score before | After | Scored higher | Scored lower |", "|---|---|---|---|---|"]
    L += [f"| {name.capitalize()} | {V[name]['pasted_twice']['mean_score_before']:.1f} | {V[name]['pasted_twice']['mean_score_after']:.1f} | "
          f"{V[name]['pasted_twice']['share_scored_higher']:.0%} | {V[name]['pasted_twice']['share_scored_lower']:.0%} |" for name in V]
    L += ["", "Limits. One seed and one training part. The test part was already printed many times while the project was built. "
          "Pasting twice is one way to add length without adding content; others were not tried.", ""]
    (RESULTS / "without_length.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s. See results/without_length.md")


if __name__ == "__main__":
    main()

"""Step 4. Four checks on the chosen model: the label, the errors, three ways to fool it, and a prompt it has never seen.

    python 04_checks.py            about 1.5 minutes on 2 cores

  A. The label. Two trained raters scored every response. How well do they agree with each other? Then a fair
     contest: a model that learned from rater 1 only is compared with rater 2, the same way rater 1 is.
  B. The errors, by prompt, by how good the response was, and by whether the two raters agreed.
     "How good" is three bands of the actual score, cut at one third and two thirds of its rank within the prompt.
     Scores are whole numbers with many ties, so the bands are not equal in size.
  C. Three probes. Each changes the test responses in a way a rater would punish, and asks what the model does:
     the same text pasted twice, the words in random order, and a good answer to a different prompt.
  D. A new prompt. The model is trained on seven prompts and scored on the eighth, in turn.

Writes results/checks.json and results/checks.md.
"""
import time

import numpy as np
import pandas as pd

from common import PROMPTS, RATER_SCALE, RESULTS, SCALE, SEED, WORK, read_json, write_json
from scoring import Scorer, measures, weighted_kappa


def rater_scaled(values, prompt):
    lo = np.array([RATER_SCALE[int(p)][0] for p in prompt]); hi = np.array([RATER_SCALE[int(p)][1] for p in prompt])
    return (np.asarray(values, float) - lo) / (hi - lo)


def rater_whole(score, prompt):
    lo = np.array([RATER_SCALE[int(p)][0] for p in prompt]); hi = np.array([RATER_SCALE[int(p)][1] for p in prompt])
    return np.clip(np.rint(lo + np.clip(score, 0, 1) * (hi - lo)), lo, hi).astype(int)


def main():
    t0 = time.time()
    chosen = read_json(RESULTS / "models.json")["chosen"]
    data = pd.read_parquet(WORK / "prepared.parquet")
    train, test = (data[data["part"] == part].reset_index(drop=True) for part in ("train", "test"))
    prompt = test["prompt"].to_numpy()
    scorer = Scorer().fit(train, kinds=[chosen, "length only"])
    predicted = scorer.predict(chosen, test["text"], prompt)
    result = {"chosen": chosen, "test_responses": len(test)}

    # ---------------------------------------------------------------- A. the label
    learned_from_1 = Scorer().fit(train.assign(score=rater_scaled(train["rater_1"], train["prompt"].to_numpy())), kinds=[chosen])
    as_rater = rater_whole(learned_from_1.predict(chosen, test["text"], prompt), prompt)
    rows = []
    for p in PROMPTS:
        m = prompt == p
        r1, r2 = test.loc[m, "rater_1"].to_numpy(), test.loc[m, "rater_2"].to_numpy()
        rows.append({"prompt": p, "responses": int(m.sum()), "raters_same_score": float((r1 == r2).mean()),
                     "kappa_rater_1_against_rater_2": weighted_kappa(r1, r2, *RATER_SCALE[p]), "kappa_model_against_rater_2": weighted_kappa(as_rater[m], r2, *RATER_SCALE[p]),
                     "model_same_score_as_rater_2": float((as_rater[m] == r2).mean())})
    result["label"] = {"by_prompt": rows, "mean_kappa_rater_1_against_rater_2": float(np.mean([r["kappa_rater_1_against_rater_2"] for r in rows])),
                       "mean_kappa_model_against_rater_2": float(np.mean([r["kappa_model_against_rater_2"] for r in rows])),
                       "prompts_where_model_is_at_least_rater_1": [r["prompt"] for r in rows if r["kappa_model_against_rater_2"] >= r["kappa_rater_1_against_rater_2"]],
                       "raters_same_score_overall": float((test["rater_1"] == test["rater_2"]).mean())}
    print(f"A. the label ({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------------------- B. the errors
    error = (predicted - test["score"].to_numpy()) * 100
    agree = (test["rater_1"] == test["rater_2"]).to_numpy()
    band = np.where(test["rank_in_prompt"] < 1 / 3, "low scores", np.where(test["rank_in_prompt"] < 2 / 3, "middle scores", "high scores"))
    result["errors"] = {
        "by_prompt": [{"prompt": p, "responses": int((prompt == p).sum()), "mean_error": float(np.abs(error[prompt == p]).mean()), "mean_signed_error": float(error[prompt == p].mean())} for p in PROMPTS],
        "by_score_band": [{"band": b, "responses": int((band == b).sum()), "mean_error": float(np.abs(error[band == b]).mean()), "mean_signed_error": float(error[band == b].mean())}
                          for b in ("low scores", "middle scores", "high scores")],
        "by_rater_agreement": [{"raters": name, "responses": int(m.sum()), "mean_error": float(np.abs(error[m]).mean())} for name, m in (("gave the same score", agree), ("gave different scores", ~agree))],
        "largest_misses": int((np.abs(error) >= 30).sum())}
    print(f"B. the errors ({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------------------- C. three probes
    rng = np.random.default_rng(SEED)
    text = test["text"].to_numpy()
    twice = np.array([t + " " + t for t in text], dtype=object)
    shuffled = np.array([" ".join(rng.permutation(t.split())) for t in text], dtype=object)
    good = np.flatnonzero(test["rank_in_prompt"].to_numpy() >= 2 / 3)                        # the high-scores band, moved to another prompt
    other = np.array([rng.choice([q for q in PROMPTS if q != p]) for p in prompt[good]])
    probes = {}
    for name, changed in (("the same text pasted twice", twice), ("the words in random order", shuffled)):
        after = scorer.predict(chosen, changed, prompt)
        probes[name] = {"responses": len(text), "mean_score_before": float(predicted.mean() * 100), "mean_score_after": float(after.mean() * 100),
                        "share_scored_higher": float((after > predicted + 1e-9).mean()), "share_scored_lower": float((after < predicted - 1e-9).mean())}
    moved = scorer.predict(chosen, text[good], other)
    probes["a good answer to a different prompt"] = {"responses": int(len(good)), "mean_score_before": float(predicted[good].mean() * 100), "mean_score_after": float(moved.mean() * 100),
                                                    "share_scored_higher": float((moved > predicted[good] + 1e-9).mean()), "share_scored_lower": float((moved < predicted[good] - 1e-9).mean())}
    result["probes"] = probes
    print(f"C. the probes ({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------------------- D. a prompt it has never seen
    inside = measures(test["score"], predicted, prompt, test["final_score"])
    rows = []
    for p in PROMPTS:
        seven = Scorer().fit(train[train["prompt"] != p].reset_index(drop=True), kinds=[chosen])
        held = test[test["prompt"] == p]
        guess = seven.predict(chosen, held["text"], held["prompt"], unseen_prompt=True)
        m = measures(held["score"], guess, held["prompt"], held["final_score"])
        rows.append({"prompt": p, "responses": len(held), "kappa_trained_with_this_prompt": inside["kappa_by_prompt"][str(p)], "kappa_never_saw_this_prompt": m["kappa_by_prompt"][str(p)],
                     "mean_error_never_saw": m["mae_points"], "mean_error_trained_with": float(np.abs(error[prompt == p]).mean()),
                     "correlation_never_saw": float(np.corrcoef(guess, held["score"])[0, 1])})
        print(f"D. prompt {p} held out ({time.time() - t0:.0f} s)", flush=True)
    result["new_prompt"] = {"by_prompt": rows, "mean_kappa_trained_with": float(np.mean([r["kappa_trained_with_this_prompt"] for r in rows])),
                            "mean_kappa_never_saw": float(np.mean([r["kappa_never_saw_this_prompt"] for r in rows])),
                            "mean_error_trained_with": float(np.mean([r["mean_error_trained_with"] for r in rows])), "mean_error_never_saw": float(np.mean([r["mean_error_never_saw"] for r in rows]))}
    result["seconds"] = round(time.time() - t0)
    write_json(RESULTS / "checks.json", result)
    write_report(result)
    print(f"done in {time.time() - t0:.0f} s. See results/checks.md")


def write_report(r):
    lab, err, new = r["label"], r["errors"], r["new_prompt"]
    L = ["# Step 4: four checks on the chosen model", "", f"The chosen model is the {r['chosen']}. Everything here is on the {r['test_responses']:,} test responses.", "",
         "## A. The label: how good is the score the model learns from?", "",
         f"Two trained raters scored every response. They gave the same score on {lab['raters_same_score_overall']:.1%} of the test responses. "
         "Agreement is measured with quadratic weighted kappa: 1 is perfect, 0 is chance.", "",
         "A fair contest: a model that learned from rater 1 alone is compared with rater 2, the same way rater 1 is.", "",
         "| Prompt | Responses | Raters gave the same score | Rater 1 against rater 2 | Model against rater 2 |", "|---|---|---|---|---|"]
    L += [f"| {x['prompt']} | {x['responses']} | {x['raters_same_score']:.1%} | {x['kappa_rater_1_against_rater_2']:.3f} | {x['kappa_model_against_rater_2']:.3f} |" for x in lab["by_prompt"]]
    L += [f"| Mean | | | {lab['mean_kappa_rater_1_against_rater_2']:.3f} | {lab['mean_kappa_model_against_rater_2']:.3f} |", "",
          f"The model reaches rater 1's level of agreement with rater 2 on {len(lab['prompts_where_model_is_at_least_rater_1'])} of 8 prompts. "
          "No model can be shown to be better than the raters by this measure: where the raters disagree, the data cannot say who is right.", "",
          "## B. Where the errors are", "", "Error in points of 100. Signed error is predicted minus actual: above zero means the model scored too high.", "",
          "| Prompt | Responses | Mean error | Mean signed error |", "|---|---|---|---|"]
    L += [f"| {x['prompt']} | {x['responses']} | {x['mean_error']:.2f} | {x['mean_signed_error']:+.2f} |" for x in err["by_prompt"]]
    L += ["", "Three bands of the actual score, cut at one third and two thirds of its rank within the prompt. Scores are whole numbers with many ties, so the bands are not equal in size: "
          "a response is in the high band when at least two thirds of the responses to its prompt scored lower.", "",
          "| Actual score, within its prompt | Responses | Mean error | Mean signed error |", "|---|---|---|---|"]
    L += [f"| {x['band'].capitalize()} | {x['responses']} | {x['mean_error']:.2f} | {x['mean_signed_error']:+.2f} |" for x in err["by_score_band"]]
    L += ["", "| The two raters | Responses | Mean error |", "|---|---|---|"]
    L += [f"| {x['raters'].capitalize()} | {x['responses']} | {x['mean_error']:.2f} |" for x in err["by_rater_agreement"]]
    L += ["", f"{err['largest_misses']} test responses are missed by 30 points or more.", "",
          "## C. Three ways to fool it", "", "Each probe changes the test responses in a way a human rater would punish. Scores are in points of 100. The third probe uses the test responses in the high band.", "",
          "| Probe | Responses | Mean predicted score before | After | Scored higher | Scored lower |", "|---|---|---|---|---|---|"]
    L += [f"| {name.capitalize()} | {x['responses']:,} | {x['mean_score_before']:.1f} | {x['mean_score_after']:.1f} | {x['share_scored_higher']:.0%} | {x['share_scored_lower']:.0%} |" for name, x in r["probes"].items()]
    L += ["", "## D. A prompt it has never seen", "",
          "The model is trained on seven prompts and scored on the eighth. A new prompt has no top and bottom responses to compare with, so the three content features are switched off, "
          "and the other features are scaled by the new prompt's own responses.", "",
          "| Prompt held out | Responses | Agreement when trained with it | Agreement when it never saw it | Mean error, trained with | Mean error, never saw | Correlation, never saw |", "|---|---|---|---|---|---|---|"]
    L += [f"| {x['prompt']} | {x['responses']} | {x['kappa_trained_with_this_prompt']:.3f} | {x['kappa_never_saw_this_prompt']:.3f} | {x['mean_error_trained_with']:.2f} | {x['mean_error_never_saw']:.2f} | {x['correlation_never_saw']:.2f} |" for x in new["by_prompt"]]
    L += [f"| Mean | | {new['mean_kappa_trained_with']:.3f} | {new['mean_kappa_never_saw']:.3f} | {new['mean_error_trained_with']:.2f} | {new['mean_error_never_saw']:.2f} | |", ""]
    (RESULTS / "checks.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

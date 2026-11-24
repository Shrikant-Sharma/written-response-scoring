"""The measures, the models and the one object that turns text into a predicted score."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge

from common import PROMPTS, SCALE, SEED
from features import FEATURES, WithinPrompt, fit_with_out_of_fold

MODELS = ["prompt average", "length only", "words only", "ridge regression", "random forest", "boosted trees"]
USES_FEATURES = ["ridge regression", "random forest", "boosted trees"]


# -------------------------------------------------------------------- measures
def to_final(score, prompt):
    """A 0 to 1 score back on the prompt's own scale, as a whole number."""
    prompt = np.asarray(prompt)
    lo = np.array([SCALE[int(p)][0] for p in prompt]); hi = np.array([SCALE[int(p)][1] for p in prompt])
    return np.clip(np.rint(lo + np.clip(score, 0, 1) * (hi - lo)), lo, hi).astype(int)


def weighted_kappa(a, b, lowest, highest):
    """Quadratic weighted kappa: agreement between two sets of whole-number scores, beyond chance. 1 is perfect, 0 is chance.
    A miss by two points costs four times a miss by one."""
    a, b = np.asarray(a, dtype=int) - lowest, np.asarray(b, dtype=int) - lowest
    n = highest - lowest + 1
    seen = np.zeros((n, n))
    np.add.at(seen, (a, b), 1)
    expected = np.outer(seen.sum(axis=1), seen.sum(axis=0)) / seen.sum()
    cost = (np.subtract.outer(np.arange(n), np.arange(n)) ** 2) / (n - 1) ** 2
    return 1.0 - (cost * seen).sum() / max((cost * expected).sum(), 1e-12)


def measures(score, predicted, prompt, final):
    """Mean absolute error in points (0 to 100), R squared, exact and within-one agreement on the prompt's scale, and the mean kappa over prompts."""
    score, predicted, prompt, final = np.asarray(score, float), np.clip(np.asarray(predicted, float), 0, 1), np.asarray(prompt), np.asarray(final, int)
    back = to_final(predicted, prompt)
    kappas = {int(p): weighted_kappa(back[prompt == p], final[prompt == p], *SCALE[int(p)]) for p in np.unique(prompt)}
    return {"mae_points": float(np.abs(score - predicted).mean() * 100), "r2": float(1 - ((score - predicted) ** 2).sum() / ((score - score.mean()) ** 2).sum()),
            "exact": float((back == final).mean()), "within_one": float((np.abs(back - final) <= 1).mean()),
            "kappa_mean": float(np.mean(list(kappas.values()))), "kappa_by_prompt": {str(k): float(v) for k, v in kappas.items()}, "responses": int(len(score))}


def paired_interval(score, predicted_a, predicted_b, draws=2000, seed=SEED):
    """How much lower is the error of b than of a? Mean absolute error in points, with a 95% interval from redrawing responses."""
    gain = (np.abs(np.asarray(score) - np.clip(predicted_a, 0, 1)) - np.abs(np.asarray(score) - np.clip(predicted_b, 0, 1))) * 100
    rng = np.random.default_rng(seed)
    means = gain[rng.integers(0, len(gain), size=(draws, len(gain)))].mean(axis=1)
    return {"gain_points": float(gain.mean()), "low": float(np.quantile(means, 0.025)), "high": float(np.quantile(means, 0.975))}


# -------------------------------------------------------------------- the design and the models
def one_hot(prompt):
    prompt = np.asarray(prompt)
    return pd.DataFrame({f"prompt_{p}": (prompt == p).astype(float) for p in PROMPTS})


class Scorer:
    """Features, their within-prompt scaling, and every model, fitted on one set of responses."""

    def fit(self, frame, kinds=MODELS):
        """frame needs text, prompt, score and fold."""
        text, prompt, score = frame["text"].to_numpy(), frame["prompt"].to_numpy(), frame["score"].to_numpy()
        self.features, own = fit_with_out_of_fold(text, prompt, score, frame["fold"].to_numpy())
        self.scale = WithinPrompt().fit(own, prompt)
        Z = self.scale.transform(own, prompt).reset_index(drop=True)
        design = pd.concat([Z, one_hot(prompt)], axis=1)
        self.prompt_mean = {int(p): float(score[prompt == p].mean()) for p in np.unique(prompt)}
        self.overall_mean = float(score.mean())
        self.models = {}
        if "length only" in kinds:
            self.models["length only"] = LinearRegression().fit(design[["log_words"] + list(one_hot(prompt).columns)], score)
        if "words only" in kinds:
            self.words = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=3, max_features=20000, token_pattern=r"[A-Za-z']{2,}").fit(text)
            self.models["words only"] = {int(p): Ridge(alpha=1.0).fit(self.words.transform(text[prompt == p]), score[prompt == p]) for p in np.unique(prompt)}
        if "ridge regression" in kinds:
            self.models["ridge regression"] = Ridge(alpha=10.0).fit(design, score)
        if "random forest" in kinds:
            self.models["random forest"] = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=2, random_state=SEED).fit(design, score)
        if "boosted trees" in kinds:
            self.models["boosted trees"] = HistGradientBoostingRegressor(learning_rate=0.06, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                                                                        early_stopping=False, random_state=SEED).fit(design, score)
        return self

    def design(self, text, prompt, unseen_prompt=False):
        raw = self.features.transform(text, prompt) if not unseen_prompt else self.raw_for_unseen(text, prompt)
        Z = self.scale.transform(raw, prompt, unseen_prompt=unseen_prompt).reset_index(drop=True)
        return pd.concat([Z, one_hot(prompt)], axis=1)

    def raw_for_unseen(self, text, prompt):
        """A prompt that was not in training has no top and bottom responses to compare with, so its three content features are set to zero."""
        from features import surface
        rows = []
        for t in text:
            f, words = surface(t)
            f = dict(f)
            f["uncommon_word_share"] = sum(w not in self.features.common for w in words) / max(len(words), 1)
            f.update(like_top_responses=0.0, like_bottom_responses=0.0, top_minus_bottom=0.0)
            rows.append(f)
        return pd.DataFrame(rows)[FEATURES]

    def predict(self, kind, text, prompt, unseen_prompt=False):
        text, prompt = np.asarray(list(text), dtype=object), np.asarray(prompt)
        if kind == "prompt average":
            return np.array([self.prompt_mean.get(int(p), self.overall_mean) for p in prompt])
        if kind == "words only":
            out = np.zeros(len(text))
            for p in np.unique(prompt):
                out[prompt == p] = self.models[kind][int(p)].predict(self.words.transform(text[prompt == p]))
            return np.clip(out, 0, 1)
        design = self.design(text, prompt, unseen_prompt)
        if unseen_prompt:
            design[[c for c in design.columns if c.startswith("prompt_")]] = 0.0     # no prompt of its own to switch on
        columns = ["log_words"] + [c for c in design.columns if c.startswith("prompt_")] if kind == "length only" else list(design.columns)
        return np.clip(self.models[kind].predict(design[columns]), 0, 1)

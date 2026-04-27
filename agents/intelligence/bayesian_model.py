from scipy.stats import beta as beta_dist

class BayesianSignalModel:
    """
    Beta-Binomial Bayesian model.
    Starts uninformed at Beta(1,1) — 50/50.
    Updates as signals are confirmed or rejected.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta  = beta

    def update(self, signal_was_correct: bool):
        if signal_was_correct:
            self.alpha += 1
        else:
            self.beta += 1

    def probability(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, ci: float = 0.95):
        return beta_dist.interval(ci, self.alpha, self.beta)

    def summary(self) -> dict:
        p = self.probability()
        lo, hi = self.credible_interval()
        return {
            "probability":  round(p, 4),
            "alpha":        self.alpha,
            "beta":         self.beta,
            "ci_low":       round(lo, 4),
            "ci_high":      round(hi, 4),
            "confidence":   "HIGH" if p > 0.7 else "MEDIUM" if p > 0.4 else "LOW"
        }


if __name__ == "__main__":
    model = BayesianSignalModel()
    print("--- BAYESIAN MODEL SIMULATION ---")
    print(f"Start : {model.summary()}")

    # simulate 10 rounds of signal feedback
    feedback = [True, False, True, True, True, False, True, True, False, True]
    for i, correct in enumerate(feedback):
        model.update(correct)
        s = model.summary()
        print(f"Round {i+1} ({'✅' if correct else '❌'}) → P={s['probability']} [{s['confidence']}] CI=({s['ci_low']}, {s['ci_high']})")

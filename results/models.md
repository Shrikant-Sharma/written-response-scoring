# Step 3: six ways to predict the score

7,797 responses to learn from, 2,571 to choose on, 2,604 for the test. The score is on a 0 to 1 scale within its prompt; errors are shown in points of 100.

## Cross-validation on the training part (five folds)

| Method | Mean error, points of 100 | Spread over the folds | R squared | Same final score | Agreement (kappa) |
|---|---|---|---|---|---|
| Prompt average | 17.63 | 0.40 | 0.095 | 32.3% | 0.000 |
| Length only | 12.05 | 0.30 | 0.536 | 47.7% | 0.624 |
| Words only | 11.29 | 0.29 | 0.604 | 48.4% | 0.610 |
| Ridge regression | 10.94 | 0.18 | 0.625 | 50.5% | 0.709 |
| Random forest | 9.83 | 0.21 | 0.686 | 55.1% | 0.753 |
| Boosted trees | 9.92 | 0.22 | 0.686 | 54.6% | 0.755 |

Best of the three feature models in cross-validation: random forest.

## The tuning part

| Method | Mean error, points of 100 | R squared | Same final score | Within one point | Agreement (kappa) |
|---|---|---|---|---|---|
| Prompt average | 17.82 | 0.112 | 29.8% | 76.7% | 0.000 |
| Length only | 12.22 | 0.533 | 46.3% | 83.6% | 0.604 |
| Words only | 11.26 | 0.607 | 47.8% | 85.8% | 0.626 |
| Ridge regression | 11.03 | 0.621 | 50.8% | 84.1% | 0.698 |
| Random forest | 9.83 | 0.673 | 55.7% | 86.5% | 0.762 |
| Boosted trees | 10.00 | 0.671 | 55.3% | 86.4% | 0.754 |

Does each step pay for itself? Cut in mean error, in points, with a 95% interval from redrawing responses 2,000 times.

| Step | Cut in mean error | 95% interval |
|---|---|---|
| Length only over prompt average | +5.59 | +5.18 to +6.02 |
| Ridge regression over length only | +1.19 | +0.95 to +1.42 |
| Random forest over ridge regression | +1.20 | +1.00 to +1.41 |
| Random forest over words only | +1.42 | +1.11 to +1.74 |

Chosen: random forest.

## The test part

| Method | Mean error, points of 100 | R squared | Same final score | Within one point | Agreement (kappa) |
|---|---|---|---|---|---|
| Prompt average | 17.75 | 0.087 | 31.0% | 75.7% | 0.000 |
| Length only | 12.16 | 0.531 | 46.8% | 84.2% | 0.622 |
| Words only | 11.20 | 0.598 | 48.7% | 85.9% | 0.619 |
| Ridge regression | 11.11 | 0.611 | 49.8% | 85.1% | 0.690 |
| Random forest | 9.88 | 0.670 | 55.1% | 86.6% | 0.746 |
| Boosted trees | 10.03 | 0.672 | 54.1% | 86.9% | 0.746 |

Against the length-only baseline, the chosen model gives the same final score as the human raters on 55.1% of test responses, up from 46.8%. That is +8.3 percentage points, or +18% in relative terms: two ways of saying the same result, and they are not interchangeable. Mean error falls from 12.16 to 9.88 points (+2.28; 95% interval +2.00 to +2.57).

Agreement by prompt on the test part (quadratic weighted kappa against the final score):

| Prompt | Score scale | Length only | Chosen model |
|---|---|---|---|
| 1 | 2 to 12 | 0.818 | 0.838 |
| 2 | 1 to 6 | 0.633 | 0.666 |
| 3 | 0 to 3 | 0.444 | 0.666 |
| 4 | 0 to 3 | 0.585 | 0.763 |
| 5 | 0 to 4 | 0.676 | 0.765 |
| 6 | 0 to 4 | 0.654 | 0.781 |
| 7 | 0 to 30 | 0.696 | 0.786 |
| 8 | 0 to 60 | 0.467 | 0.705 |

## What the chosen model leans on

Each feature is shuffled in turn on the tuning part. The rise in mean error says how much the model needs it.

| Feature | Rise in mean error, points |
|---|---|
| how much more its wording resembles the top quarter than the bottom quarter | 2.09 |
| number of different words | 1.61 |
| number of words | 1.10 |
| digits per 100 words | 1.08 |
| number of words, on a log scale | 1.02 |
| similarity of its wording to the top quarter of training responses on the same prompt | 0.21 |
| number of sentences | 0.14 |
| question and exclamation marks per 100 words | 0.11 |

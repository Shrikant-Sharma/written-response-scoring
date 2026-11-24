# Step 4: four checks on the chosen model

The chosen model is the random forest. Everything here is on the 2,604 test responses.

## A. The label: how good is the score the model learns from?

Two trained raters scored every response. They gave the same score on 61.7% of the test responses. Agreement is measured with quadratic weighted kappa: 1 is perfect, 0 is chance.

A fair contest: a model that learned from rater 1 alone is compared with rater 2, the same way rater 1 is.

| Prompt | Responses | Raters gave the same score | Rater 1 against rater 2 | Model against rater 2 |
|---|---|---|---|---|
| 1 | 354 | 65.8% | 0.739 | 0.759 |
| 2 | 350 | 76.6% | 0.777 | 0.696 |
| 3 | 355 | 74.1% | 0.748 | 0.595 |
| 4 | 354 | 77.1% | 0.846 | 0.699 |
| 5 | 382 | 56.5% | 0.716 | 0.758 |
| 6 | 342 | 63.5% | 0.809 | 0.744 |
| 7 | 314 | 28.0% | 0.702 | 0.698 |
| 8 | 153 | 32.0% | 0.684 | 0.556 |
| Mean | | | 0.753 | 0.688 |

The model reaches rater 1's level of agreement with rater 2 on 2 of 8 prompts. No model can be shown to be better than the raters by this measure: where the raters disagree, the data cannot say who is right.

## B. Where the errors are

Error in points of 100. Signed error is predicted minus actual: above zero means the model scored too high.

| Prompt | Responses | Mean error | Mean signed error |
|---|---|---|---|
| 1 | 354 | 6.06 | -1.25 |
| 2 | 350 | 7.17 | -0.02 |
| 3 | 355 | 14.37 | -0.34 |
| 4 | 354 | 13.95 | +0.99 |
| 5 | 382 | 10.61 | +1.36 |
| 6 | 342 | 11.32 | -2.51 |
| 7 | 314 | 7.09 | +0.44 |
| 8 | 153 | 5.73 | -1.17 |

Three bands of the actual score, cut at one third and two thirds of its rank within the prompt. Scores are whole numbers with many ties, so the bands are not equal in size: a response is in the high band when at least two thirds of the responses to its prompt scored lower.

| Actual score, within its prompt | Responses | Mean error | Mean signed error |
|---|---|---|---|
| Low scores | 1225 | 10.09 | +5.84 |
| Middle scores | 893 | 8.47 | -2.55 |
| High scores | 486 | 11.92 | -11.25 |

| The two raters | Responses | Mean error |
|---|---|---|
| Gave the same score | 1607 | 9.67 |
| Gave different scores | 997 | 10.21 |

84 test responses are missed by 30 points or more.

## C. Three ways to fool it

Each probe changes the test responses in a way a human rater would punish. Scores are in points of 100. The third probe uses the test responses in the high band.

| Probe | Responses | Mean predicted score before | After | Scored higher | Scored lower |
|---|---|---|---|---|---|
| The same text pasted twice | 2,604 | 57.6 | 65.0 | 93% | 7% |
| The words in random order | 2,604 | 57.6 | 57.3 | 36% | 64% |
| A good answer to a different prompt | 486 | 78.8 | 62.8 | 28% | 72% |

## D. A prompt it has never seen

The model is trained on seven prompts and scored on the eighth. A new prompt has no top and bottom responses to compare with, so the three content features are switched off, and the other features are scaled by the new prompt's own responses.

| Prompt held out | Responses | Agreement when trained with it | Agreement when it never saw it | Mean error, trained with | Mean error, never saw | Correlation, never saw |
|---|---|---|---|---|---|---|
| 1 | 354 | 0.838 | 0.667 | 6.06 | 8.84 | 0.84 |
| 2 | 350 | 0.666 | 0.490 | 7.17 | 12.41 | 0.71 |
| 3 | 355 | 0.666 | 0.377 | 14.37 | 17.32 | 0.66 |
| 4 | 354 | 0.763 | 0.389 | 13.95 | 20.35 | 0.75 |
| 5 | 382 | 0.765 | 0.669 | 10.61 | 12.32 | 0.82 |
| 6 | 342 | 0.781 | 0.623 | 11.32 | 14.60 | 0.77 |
| 7 | 314 | 0.786 | 0.549 | 7.09 | 11.90 | 0.71 |
| 8 | 153 | 0.705 | 0.540 | 5.73 | 9.86 | 0.58 |
| Mean | | 0.746 | 0.538 | 9.54 | 13.45 | |


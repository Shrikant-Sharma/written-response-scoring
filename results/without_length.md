# Extension, part 2: the forest without its length features

Three forests on the same training part, with the same settings and seed. Errors are in points of 100.

The full forest repeats step 3 and the paste-twice probe of step 4.

| Forest | Features | Taken out |
|---|---|---|
| Full | 20 | nothing |
| No size counts | 17 | words, log_words, sentences |
| No length | 16 | words, log_words, sentences, distinct_words |

## Accuracy on the tune part

Length alone, from step 3: 12.22.

| Forest | Mean error | R squared | Same final score | Agreement (kappa) | Cost against the full forest | 95% interval |
|---|---|---|---|---|---|---|
| Full | 9.83 | 0.673 | 55.7% | 0.762 |  |  |
| No size counts | 9.99 | 0.669 | 55.3% | 0.759 | +0.16 | +0.07 to +0.25 |
| No length | 10.99 | 0.625 | 50.8% | 0.705 | +1.16 | +0.93 to +1.37 |

## Accuracy on the test part

Length alone, from step 3: 12.16.

| Forest | Mean error | R squared | Same final score | Agreement (kappa) | Cost against the full forest | 95% interval |
|---|---|---|---|---|---|---|
| Full | 9.88 | 0.670 | 55.1% | 0.746 |  |  |
| No size counts | 10.03 | 0.665 | 54.1% | 0.738 | +0.15 | +0.06 to +0.23 |
| No length | 10.98 | 0.615 | 49.6% | 0.699 | +1.10 | +0.89 to +1.31 |

The cost is how many points of mean error the full forest saves over that forest, with a 95% interval from redrawing responses 2,000 times.

## The same text pasted twice, on the test part

| Forest | Mean predicted score before | After | Scored higher | Scored lower |
|---|---|---|---|---|
| Full | 57.6 | 65.0 | 93% | 7% |
| No size counts | 57.6 | 56.3 | 30% | 70% |
| No length | 57.5 | 43.3 | 2% | 98% |

Limits. One seed and one training part. The test part was already printed many times while the project was built. Pasting twice is one way to add length without adding content; others were not tried.


# Extension, part 1: the seed check

The forest is grown from 5 seeds on the same training part. Ridge regression and the boosted trees involve no random draw and are fitted once. Errors and gains are in points of 100.

The first seed is the project's own. Its row repeats step 3.

## The tune part

Ridge regression: 11.03. Boosted trees: 10.00.

| Seed | Forest, mean error | Gain over ridge | 95% interval | Gain over boosted trees | 95% interval |
|---|---|---|---|---|---|
| 20261122 | 9.83 | +1.20 | +1.00 to +1.41 | +0.16 | +0.05 to +0.29 |
| 1 | 9.86 | +1.16 | +0.96 to +1.38 | +0.13 | +0.02 to +0.25 |
| 2 | 9.82 | +1.21 | +1.01 to +1.42 | +0.17 | +0.06 to +0.30 |
| 3 | 9.79 | +1.23 | +1.04 to +1.45 | +0.20 | +0.09 to +0.33 |
| 4 | 9.83 | +1.20 | +1.00 to +1.41 | +0.17 | +0.05 to +0.29 |

Over ridge regression the gain runs from 1.16 to 1.23 points across the seeds, and the lowest end of any interval is +0.96. Over the boosted trees it runs from +0.13 to +0.20, and the interval is clear of zero for 5 of 5 seeds.

## The test part

Ridge regression: 11.11. Boosted trees: 10.03.

| Seed | Forest, mean error | Gain over ridge | 95% interval | Gain over boosted trees | 95% interval |
|---|---|---|---|---|---|
| 20261122 | 9.88 | +1.23 | +1.01 to +1.44 | +0.15 | +0.03 to +0.26 |
| 1 | 9.85 | +1.26 | +1.04 to +1.48 | +0.18 | +0.06 to +0.29 |
| 2 | 9.85 | +1.26 | +1.04 to +1.47 | +0.18 | +0.07 to +0.30 |
| 3 | 9.84 | +1.27 | +1.05 to +1.49 | +0.19 | +0.07 to +0.31 |
| 4 | 9.87 | +1.24 | +1.03 to +1.46 | +0.16 | +0.04 to +0.28 |

Over ridge regression the gain runs from 1.23 to 1.27 points across the seeds, and the lowest end of any interval is +1.01. Over the boosted trees it runs from +0.15 to +0.19, and the interval is clear of zero for 5 of 5 seeds.

What this does not cover: the training responses themselves are the same in every row. Only the forest's own random draws change.


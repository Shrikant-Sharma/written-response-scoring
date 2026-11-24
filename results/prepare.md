# Step 2: the data, prepared in SQL

12,976 responses in the file. 4 response texts appear more than once (4 extra copies); both ways of finding them agree. 2 of those texts were given two different final scores. The first copy of each is kept: 12,972 responses.

Split by a fingerprint of the response id: 7,797 to learn from, 2,571 to make choices on, 2,604 held back for the end.

## The two human raters

| Prompt | Responses | Same score | Within one point | Largest gap | Correlation |
|---|---|---|---|---|---|
| 1 | 1,783 | 65.3% | 98.8% | 2 | 0.72 |
| 2 | 1,800 | 78.3% | 99.8% | 2 | 0.81 |
| 3 | 1,724 | 74.8% | 99.7% | 3 | 0.77 |
| 4 | 1,768 | 77.1% | 99.9% | 2 | 0.85 |
| 5 | 1,805 | 58.0% | 98.0% | 3 | 0.75 |
| 6 | 1,800 | 62.3% | 98.5% | 3 | 0.78 |
| 7 | 1,569 | 29.2% | 60.7% | 9 | 0.72 |
| 8 | 723 | 27.8% | 48.1% | 9 | 0.63 |

The raters of prompts 7 and 8 score on wider scales, so an exact match is rarer there.

## How the final score follows from the two raters

It differs by prompt. The shares below are the responses whose final score equals the sum of the two raters, the higher of the two, or rater 1's score.

| Prompt | Published scale | Final scores in the file | One rater's scores in the file | Final is the sum | Final is the higher | Final is rater 1's |
|---|---|---|---|---|---|---|
| 1 | 2 to 12 | 2 to 12 | 1 to 6 | 100.0% | 0.0% | 0.0% |
| 2 | 1 to 6 | 1 to 6 | 1 to 6 | 0.0% | 88.1% | 100.0% |
| 3 | 0 to 3 | 0 to 3 | 0 to 3 | 2.5% | 99.7% | 88.9% |
| 4 | 0 to 3 | 0 to 3 | 0 to 3 | 18.1% | 99.9% | 88.8% |
| 5 | 0 to 4 | 0 to 4 | 0 to 4 | 4.0% | 96.6% | 78.0% |
| 6 | 0 to 4 | 0 to 4 | 0 to 4 | 3.6% | 96.9% | 80.6% |
| 7 | 0 to 30 | 2 to 24 | 0 to 12 | 100.0% | 0.2% | 0.1% |
| 8 | 0 to 60 | 10 to 60 | 5 to 30 | 84.2% | 0.0% | 0.0% |

The published scales are used as given. Where the file does not reach the ends of a published scale (prompt 7 stops at 24 of 30, prompt 8 starts at 10), a point of 100 is a point of the published scale, not of the range seen in the file.


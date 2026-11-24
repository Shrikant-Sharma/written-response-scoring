# Decisions

A log of the choices in this project: the situation, the options, the choice, the reason, and what happened.
Entries 1 to 6 are choices that were already in the design. I ran each one on November 24, 2026, looked at the options, and kept it. Entries 7 to 9 are mine.

## 1. Which range to rescale the scores by

- **Date:** November 24, 2026
- **Situation:** The eight prompts are scored out of 3, 4, 6, 12, 30 and 60. One model needs one scale.
- **Options:**
  - The published scale of each prompt. A point of 100 is a point of the real scale, and nothing is learned from the data. Cost: part of the scale is never used on two prompts.
  - The range seen in the file. Every prompt spans 0 to 1. Cost: the range is learned from data, and a point means something different on each prompt.
- **Choice:** The published scale. Kept.
- **Reason:** It cannot leak, and an error of one point keeps one meaning.
- **What happened:** On my run, prompt 7 spans 0.067 to 0.8 and prompt 8 starts at 0.167. All other prompts span 0 to 1.

## 2. How to split the responses

- **Date:** November 24, 2026
- **Situation:** The data needs a train, tune and test part, and five folds inside train.
- **Options:**
  - A fingerprint (MD5) of the response id, in SQL. Same split on any machine and any row order. Cost: sizes are close to the target, not exact, and not balanced by prompt on purpose.
  - A random split with a seed. Exact sizes. Cost: it depends on row order and library version.
  - Stratified by prompt and score. Same score mix in each part. Cost: it reads the score to make the split.
  - By writer. The right way when one person has several rows. Not possible: the file has no writer id.
- **Choice:** The fingerprint of the id. Kept.
- **Reason:** The split never depends on row order, so anyone can rebuild it.
- **What happened:** On my run: 7,797 train (60.1%), 2,571 tune (19.8%), 2,604 test (20.1%). Mean scores 0.580, 0.585 and 0.579, so no part got easier responses. Folds of 1,551, 1,537, 1,610, 1,510 and 1,589.

## 3. Three parts: 60, 20, 20

- **Date:** November 24, 2026
- **Situation:** A model has to be chosen before the test part is scored.
- **Options:**
  - Train, tune and test. The tune part answers one question with an interval before the test. Cost: 20% of the data is not used for learning.
  - Train and test only, choosing by cross-validation alone. More data to learn from. Cost: no separate part to check the choice on.
- **Choice:** 60, 20, 20. Kept.
- **Reason:** The choice between models gets its own data, apart from the test.
- **What happened:** On my run the tune part answered its one question: the forest cut the mean error by 1.20 points over ridge regression, with a 95% interval from 1.00 to 1.41.

## 4. What the model is trained to predict

- **Date:** November 24, 2026
- **Situation:** Each response has two raters' scores and a final score. The final score is formed differently by prompt: the sum on prompts 1 and 7 and on 84% of prompt 8, rater 1's score on prompt 2, the higher of the two on 97% to 100% of prompts 3 to 6.
- **Options:**
  - The final score on file. It is the score of record. Cost: "the label" means slightly different things across prompts.
  - One rater's score. One meaning. Cost: it drops the second opinion.
  - Both raters, separately. Shows the disagreement. Cost: twice the modelling.
- **Choice:** The final score for the main result, and one rater's score for the label check. Kept.
- **Reason:** A scoring model has to match the score of record. The label check then asks the fair question with one rater against the other.
- **What happened:** On my run the two raters gave the same score on 62.1% of all responses and 61.7% of test responses. Four texts were filed twice, and two of them got two different final scores (one got a 3 and a 1 on a 0 to 3 scale).

## 5. How the model is chosen

- **Date:** November 24, 2026
- **Situation:** Three models use the 20 features: ridge regression, a random forest and boosted trees. One has to be picked without looking at the test part.
- **Options:**
  - A rule fixed in the code: take the best in five-fold cross-validation, and keep it only if its gain over ridge regression on the tune part has a 95% interval above zero. Otherwise keep ridge. Cost: the interval redraws responses but holds the trained models fixed, so it leaves out the luck of the training seed.
  - Pick the lowest error on the test part. Cost: the test part would then be part of the choice.
  - Always take the simplest model. Cost: it gives up a real gain.
- **Choice:** The fixed rule. Kept.
- **Reason:** The test part plays no role in the choice, and the simpler model wins unless the complex one earns its place.
- **What happened:** On my run, cross-validation put the forest at 9.83 points and the boosted trees at 9.92, with a spread of about 0.2 between folds, so those two are a tie. The forest came first and its gain over ridge was 1.20 points (1.00 to 1.41). It was chosen. On the test part: 9.88 points, 55.1% exact, kappa 0.746, against 12.16, 46.8% and 0.622 for length alone.

## 6. Model settings

- **Date:** November 24, 2026
- **Situation:** Each method has settings: 300 trees with at least 5 responses per leaf and half the features per split for the forest, 300 rounds for the boosted trees, a penalty of 10 for ridge.
- **Options:**
  - Written by hand, with no search. Small and fast, and nothing is tuned on the data. Cost: each model might do a little better, and nobody knows which would win after tuning.
  - A grid or random search inside cross-validation. Cost: more code and time, and one more thing that can leak if done carelessly.
- **Choice:** Written by hand. Kept.
- **Reason:** The project is about validation and the limits of the label, not about the last tenth of a point.
- **What happened:** I do not claim tuned settings, and I do not claim the forest beats the boosted trees.

## 7. Before quoting the gain: the seed check

- **Date:** November 24, 2026
- **Situation:** The interval for the forest's gain over ridge regression redraws the responses but holds the trained models fixed. A forest is grown with random draws, so the gain rested on one training draw.
- **Options:**
  - Quote the gain from one seed. Cost: it leaves out the luck of the seed.
  - Grow the forest from five seeds and quote the gain as a range. Cost: one new file and about a minute.
- **Choice:** Five seeds, in `06_seed_check.py`.
- **Reason:** A gain of one model over another should hold when the model is trained again.
- **What happened:** The first seed repeats step 3. Over ridge regression the gain runs from 1.16 to 1.23 points on the tune part and from 1.23 to 1.27 on the test part, and the lowest end of any interval is 0.96. Not covered: the training responses are the same in every row.
- **Also found:** Over the boosted trees the forest is ahead by 0.13 to 0.20 points, with the interval clear of zero for 5 of 5 seeds on both parts. Cross-validation had them level within the spread of the folds. Paired, there is a small, steady edge. Neither model was tuned, so I do not claim that forests beat boosting in general.

## 8. My change: the forest without its length features

- **Date:** November 24, 2026
- **Situation:** Length alone is a strong baseline, and pasting a response twice raised its score for 93% of test responses. I wanted to know how much of the model is length, and whether it is harder to fool without it.
- **Options:**
  - Take the length features away. It tests the model's weakest point.
  - Add a seventh method, a single tree, to show what bagging buys. A clean result, and not a surprising one.
  - Measure how many scored responses a new prompt needs. The most new code, and one prompt may not speak for the others.
- **Choice:** Take the length features away, in `07_without_length.py`. Three forests with the same settings and seed: all 20 features, 17 without the three size counts (words, their log, sentences), and 16 without those and the number of different words.
- **What I expected:** No prediction was written down before the run.
- **What happened, on the test part:**
  - 17 features: mean error 10.03 against 9.88, a cost of 0.15 points (0.06 to 0.23). Kappa 0.738 against 0.746. Pasted twice, 30% of responses score higher, down from 93%, and the mean score goes from 57.6 to 56.3.
  - 16 features: mean error 10.98, a cost of 1.10 points (0.89 to 1.31). Kappa 0.699. Still ahead of length alone at 12.16. Pasted twice, 98% score lower and the mean falls from 57.5 to 43.3.
  - The tune part shows the same: costs of 0.16 and 1.16 points.
- **What I learned:** The three size counts add almost nothing to accuracy and are the whole reward for pasting. Length was also carried by the number of different words. With no length feature at all the model is pushed the other way by the same trick, through a ratio feature, so it still counts and does not read.
- **The choice that followed:** The full forest stays as the benchmark, and every reported number is its number. If the model were used, I would use the 17-feature forest: 0.15 points given up to remove most of the reward for padding.
- **Limits:** One seed, one training part, one way of adding length. The 17-feature forest was named after its test results were seen, and the test part was never sealed.

## 9. What the score may be used for

- **Date:** November 24, 2026
- **Situation:** The project measured its own limits. A decision on use has to follow from them.
- **The evidence, from my run:** A model trained on rater 1 agrees with rater 2 at kappa 0.688, where rater 1 reaches 0.753. The high band of responses is scored 11.25 points too low. On a prompt it has never seen, agreement falls from 0.746 to 0.538. It rewards length. The file says nothing about the writers, so no fairness test was possible.
- **Options:**
  - The score alone. Cost: it is below a second rater, it can be gamed, and it is hardest on the strongest responses.
  - A second opinion. A person scores every response, and the model flags the ones where it differs by a lot for another human look. Cost: it saves no scoring time.
  - Triage only. The model sorts responses into rough bands. Cost: a wrong band still delays or misroutes a response.
  - Not at all. Cost: it gives up what the model can do.
- **Choice:** A second opinion, and only that.
- **Conditions:** A person reads and scores every response. A new prompt gets its own scored responses before the model is used on it. It is not used on people in a hiring setting until a fairness test has been run on data that describes the writers. The 17-feature forest is the one used.
- **Reason:** The raters themselves disagree on 38% of responses, and one text got a 3 and a 1. A second look at the cases where a rater and the model are far apart puts the model where its errors cost least and its consistency helps most.
- **What happened:** Nothing is deployed. This is the position I take when asked.

## Smaller choices I can explain

- **What counts as a duplicate:** the same text ignoring case and spacing. Exact matching finds the same 4 on this file. Near-duplicates were not searched.
- **Which copy is kept:** the first by response id. Nothing in the file says which score is right.
- **Why duplicates are found two ways:** the second way checks the first. The script stops if they disagree.
- **Who reads the file:** pandas, because the text is in an old Windows encoding. SQL does the rest.

-- How often do the two human raters give the same score, and how far apart are they when they do not?
-- And how does the final score on file follow from their two scores? It differs by prompt.
SELECT prompt,
       count(*)                                                        AS responses,
       round(avg(CASE WHEN rater_1 = rater_2 THEN 1.0 ELSE 0.0 END), 4) AS same_score,
       round(avg(CASE WHEN abs(rater_1 - rater_2) <= 1 THEN 1.0 ELSE 0.0 END), 4) AS within_one,
       max(abs(rater_1 - rater_2))                                     AS largest_gap,
       round(corr(rater_1, rater_2), 4)                                AS correlation,
       round(avg(CASE WHEN final_score = rater_1 + rater_2 THEN 1.0 ELSE 0.0 END), 4)          AS final_is_the_sum,
       round(avg(CASE WHEN final_score = greatest(rater_1, rater_2) THEN 1.0 ELSE 0.0 END), 4) AS final_is_the_higher,
       round(avg(CASE WHEN final_score = rater_1 THEN 1.0 ELSE 0.0 END), 4)                    AS final_is_rater_1,
       min(final_score)                                                AS lowest_final,
       max(final_score)                                                AS highest_final,
       min(least(rater_1, rater_2))                                    AS lowest_rater,
       max(greatest(rater_1, rater_2))                                 AS highest_rater
FROM prepared
GROUP BY prompt
ORDER BY prompt;

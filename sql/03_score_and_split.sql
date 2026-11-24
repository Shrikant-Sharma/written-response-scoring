-- Put every prompt's score on one 0 to 1 scale, and split the responses three ways.
-- The split uses a fingerprint of the response id, so it is the same on every machine and every run.
-- A second piece of the same fingerprint numbers five folds for cross-validation.
CREATE OR REPLACE TABLE prepared AS
SELECT r.response_id, r.prompt, r.text, r.rater_1, r.rater_2, r.rater_3, r.final_score,
       (r.final_score - s.lowest) / (s.highest - s.lowest)::DOUBLE                     AS score,
       avg((r.final_score - s.lowest) / (s.highest - s.lowest)::DOUBLE)
           OVER (PARTITION BY r.prompt)                                                 AS prompt_mean_score,
       percent_rank() OVER (PARTITION BY r.prompt ORDER BY r.final_score)               AS rank_in_prompt,
       CASE WHEN bucket < 60 THEN 'train' WHEN bucket < 80 THEN 'tune' ELSE 'test' END  AS part,
       fold
FROM (SELECT *, ('0x' || substr(md5(response_id::VARCHAR), 1, 7))::BIGINT % 100 AS bucket,
                ('0x' || substr(md5(response_id::VARCHAR), 8, 7))::BIGINT % 5   AS fold
      FROM responses) r
JOIN scale s USING (prompt)
ORDER BY r.response_id;

-- The raw file as it is, one row per response. Python reads the file (its text is in an old Windows encoding)
-- and hands it over as the table file_rows.
CREATE OR REPLACE TABLE raw AS
SELECT essay_id::INTEGER        AS response_id,
       essay_set::INTEGER       AS prompt,
       essay                    AS text,
       rater1_domain1::INTEGER  AS rater_1,
       rater2_domain1::INTEGER  AS rater_2,
       rater3_domain1::INTEGER  AS rater_3,
       domain1_score::INTEGER   AS final_score
FROM file_rows;

-- The score range of each prompt, from the publisher's set descriptions.
CREATE OR REPLACE TABLE scale AS
SELECT * FROM (VALUES (1, 2, 12), (2, 1, 6), (3, 0, 3), (4, 0, 3), (5, 0, 4), (6, 0, 4), (7, 0, 30), (8, 0, 60)) AS t(prompt, lowest, highest);

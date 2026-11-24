-- The same response text filed twice would sit on both sides of a split, so duplicates go before the split.
-- A text key ignores case and spacing.
CREATE OR REPLACE TABLE keyed AS
SELECT *, lower(regexp_replace(trim(text), '\s+', ' ', 'g')) AS text_key
FROM raw;

-- Way 1: group and count.
CREATE OR REPLACE TABLE duplicate_groups AS
SELECT text_key, count(*) AS copies, min(response_id) AS first_id, count(DISTINCT prompt) AS prompts,
       min(final_score) AS lowest_score, max(final_score) AS highest_score
FROM keyed
GROUP BY text_key
HAVING count(*) > 1;

-- Way 2: number the copies inside each group and keep the first.
CREATE OR REPLACE TABLE numbered AS
SELECT *, row_number() OVER (PARTITION BY text_key ORDER BY response_id) AS copy_number
FROM keyed;

CREATE OR REPLACE TABLE responses AS
SELECT response_id, prompt, text, rater_1, rater_2, rater_3, final_score
FROM numbered
WHERE copy_number = 1;

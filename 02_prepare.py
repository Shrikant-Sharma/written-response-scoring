"""Step 2. Prepare the data in SQL: load, remove duplicate responses, rescale the scores, split, and measure the raters.

    python 02_prepare.py            a few seconds

The SQL is in sql/, one file per job, and runs in DuckDB (a database that lives inside the Python process).
  01_load.sql              the raw file into a table (pandas reads the file, SQL does the rest)
  02_duplicates.sql        duplicate responses found two ways, first copy kept
  03_score_and_split.sql   every prompt's score on a 0 to 1 scale; train, tune and test parts by a fingerprint of the id
  04_rater_agreement.sql   how often the two human raters agree, per prompt, and how the final score follows from their scores

Writes data/work/prepared.parquet, results/prepare.json and results/prepare.md.
"""
import time

import duckdb
import pandas as pd

from common import DATA_FILE, RAW, RESULTS, SCALE, SQL, WORK, write_json


def run(con, name):
    return con.execute((SQL / name).read_text(encoding="utf-8"))


def main():
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    file_rows = pd.read_csv(RAW / DATA_FILE, sep="\t", encoding="latin-1", dtype=str)
    con.register("file_rows", file_rows)
    run(con, "01_load.sql")
    run(con, "02_duplicates.sql")
    run(con, "03_score_and_split.sql")
    raw_rows = con.execute("SELECT count(*) FROM raw").fetchone()[0]
    groups = con.execute("SELECT count(*), coalesce(sum(copies - 1), 0), coalesce(sum(CASE WHEN lowest_score <> highest_score THEN 1 ELSE 0 END), 0) FROM duplicate_groups").fetchone()
    by_window = con.execute("SELECT count(*) FROM numbered WHERE copy_number > 1").fetchone()[0]
    assert groups[1] == by_window, "the two ways of finding duplicates disagree"
    duplicates = con.execute("SELECT first_id, copies, prompts, lowest_score, highest_score FROM duplicate_groups ORDER BY first_id").fetchall()
    parts = con.execute("SELECT prompt, part, count(*), avg(score) FROM prepared GROUP BY prompt, part ORDER BY prompt, part").fetchall()
    agreement = run(con, "04_rater_agreement.sql").fetchall()
    con.execute(f"COPY prepared TO '{(WORK / 'prepared.parquet').as_posix()}' (FORMAT parquet)")
    kept = con.execute("SELECT count(*) FROM prepared").fetchone()[0]
    result = {
        "raw_rows": raw_rows, "duplicate_groups": groups[0], "duplicate_rows_removed": int(groups[1]), "duplicate_groups_with_two_different_scores": int(groups[2]),
        "duplicates": [{"first_id": a, "copies": b, "prompts": c, "lowest_score": d, "highest_score": e} for a, b, c, d, e in duplicates],
        "kept": kept,
        "parts": [{"prompt": p, "part": part, "responses": n, "mean_score": m} for p, part, n, m in parts],
        "rater_agreement": [{"prompt": r[0], "responses": r[1], "same_score": r[2], "within_one": r[3], "largest_gap": r[4], "correlation": r[5]} for r in agreement],
        "final_score": [{"prompt": r[0], "published_scale": list(SCALE[r[0]]), "final_is_the_sum_of_the_raters": r[6], "final_is_the_higher_rater": r[7], "final_is_rater_1": r[8],
                         "lowest_final_in_file": r[9], "highest_final_in_file": r[10], "lowest_rater_score_in_file": r[11], "highest_rater_score_in_file": r[12]} for r in agreement],
    }
    write_json(RESULTS / "prepare.json", result)
    count = lambda part: sum(r["responses"] for r in result["parts"] if r["part"] == part)
    L = ["# Step 2: the data, prepared in SQL", "",
         f"{raw_rows:,} responses in the file. {groups[0]} response texts appear more than once ({int(groups[1])} extra copies); both ways of finding them agree. "
         f"{int(groups[2])} of those texts were given two different final scores. The first copy of each is kept: {kept:,} responses.", "",
         f"Split by a fingerprint of the response id: {count('train'):,} to learn from, {count('tune'):,} to make choices on, {count('test'):,} held back for the end.", "",
         "## The two human raters", "",
         "| Prompt | Responses | Same score | Within one point | Largest gap | Correlation |", "|---|---|---|---|---|---|"]
    L += [f"| {r['prompt']} | {r['responses']:,} | {r['same_score']:.1%} | {r['within_one']:.1%} | {r['largest_gap']} | {r['correlation']:.2f} |" for r in result["rater_agreement"]]
    L += ["", "The raters of prompts 7 and 8 score on wider scales, so an exact match is rarer there.", "",
          "## How the final score follows from the two raters", "",
          "It differs by prompt. The shares below are the responses whose final score equals the sum of the two raters, the higher of the two, or rater 1's score.", "",
          "| Prompt | Published scale | Final scores in the file | One rater's scores in the file | Final is the sum | Final is the higher | Final is rater 1's |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['prompt']} | {r['published_scale'][0]} to {r['published_scale'][1]} | {r['lowest_final_in_file']} to {r['highest_final_in_file']} | {r['lowest_rater_score_in_file']} to {r['highest_rater_score_in_file']} | "
          f"{r['final_is_the_sum_of_the_raters']:.1%} | {r['final_is_the_higher_rater']:.1%} | {r['final_is_rater_1']:.1%} |" for r in result["final_score"]]
    L += ["", "The published scales are used as given. Where the file does not reach the ends of a published scale (prompt 7 stops at 24 of 30, prompt 8 starts at 10), "
          "a point of 100 is a point of the published scale, not of the range seen in the file.", ""]
    (RESULTS / "prepare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"done in {time.time() - t0:.1f} s. {kept:,} responses kept. See results/prepare.md")


if __name__ == "__main__":
    main()

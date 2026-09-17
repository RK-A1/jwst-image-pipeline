# Where this left off

Last worked on 17 September 2026. This repository now holds the whole pipeline: the
Flickr ingest and embedding work from RK-A1/JWST, merged with its history, and the
labelling and dataset build that were here before. The published dataset is
rebuilt by the pipeline. From the migrated warehouse it matched the previous build
row for row, apart from two intended changes: `source_image_path`, a path on one
laptop, became `image_file`, and photo 55252854454 gained the embedding it had been
missing. A separate commit then corrected `instrument` on 115 rows.

## State

- **Warehouse:** migrated from the old pipeline's DuckDB file by
  `scripts/migrate_legacy_warehouse.py`. It holds 4,345 photos, 4,336 embeddings and
  1,077 labels. The old file in `jwst-image-pipeline/include/` was left untouched.
- **Images:** the 18 GB of originals were moved, not copied, into `include/data/images/`.
  The old repository no longer has them.
- **Flickr is ahead of the warehouse.** The key in `.env` works. On 17 September Flickr
  listed 4,402 photos against the warehouse's 4,345, so 57 are new since the last
  ingest on 7 May. The most recent are real observations. Nothing has been ingested
  since then, by decision: no API calls were made during the migration.
- **Nine legacy images are truncated** (one is empty) and marked for download again.
  The first `jwst_ingest` run fetches them.
- **Three image files have no photo row:** 52504493014, 52693571483, 53061873589.
  They are probably downloads whose insert failed in the old pipeline. They are
  harmless, and the next ingest will add rows for them if Flickr still lists them.

## Verified, and not yet verified

`jwst_dataset` and `jwst_embed` were run end to end with `airflow dags test` on the host
against the real warehouse, and all 78 tests pass on the host with torch installed.
**The DAGs have not yet run under `astro dev start`.** Docker Desktop would not start
during that session. The first thing to do is:

```bash
astro dev start          # check the duckdb_warehouse pool exists in Admin → Pools
astro dev pytest         # includes the torch test that skips on the host
```

`jwst_ingest` and `jwst_label` have only been tested against fakes, because both call
external APIs.

## Next run, in order

1. Unpause `jwst_ingest`, `jwst_embed`, `jwst_dataset`. The ingest picks up the 57 new
   photos and the 9 re-downloads, and the embed DAG follows on its own.
2. With an Anthropic key in `.env`, trigger `jwst_label`. About 57 photos at half a
   cent each is roughly $0.30. `jwst_dataset` then rebuilds and publishes.
3. Commit `include/data/labels/raw_labels.jsonl` and `include/data/dataset/`.

## The one thing that actually matters

**Nobody has checked the labels.** They are model-generated and no accuracy figure
exists. Every `jwst_dataset` run now regenerates `include/data/review/review.html`,
a sample of 100 rows stratified across subject classes. Look first at the 13 rows where
`object_name_verified` is false, and at the `galaxy` / `nebula` / `deep_field`
boundaries. When the correction rate is known, put it in the README.

## Known defects, in priority order

**`instrument` still has 11 unsupported single values.** The build now corrects
`multiple` from the caption text (`assemble.correct_instrument`), which changed 115 of
156 rows: 96 to `unknown`, 11 to NIRCam, 5 to MIRI and 3 to NIRSpec. That rule only
touches `multiple`. Seven rows say MIRI and four say NIRCam although their captions
never name them. Those values may come from text printed on the image, so they were
left alone. Check them on the review sheet before extending the rule.

**Prompt caching has never worked.** `cache_read_tokens` is 0 on all 1,077 calls. Claude
Haiku 4.5 caches only prompts of 4,096 tokens or more, and these prompts average 3,886
tokens, image included. The `cache_control` marker is harmless but does nothing. Do
not pad the prompt to reach the threshold; at this volume the saving is cents.

**`codebook.md`'s revision log stops at v2**, while the labels were produced at v6. The
changes from v3 to v6 touched modality and instrument, per the old NEXT.md, but were
never written up. Reconstruct them from `scripts/compare_label_runs.py` and the v2
backup in `include/data/labels/archive/` before the codebook changes again.

**`object_name` drifted slightly at v6.** A few rows expand or substitute names against
the codebook's own instruction to copy verbatim: `30 Doradus` became
`Tarantula Nebula`, and `WR 140` became `Wolf-Rayet 140`. `object_name_verified`
already flags them.

**`confidence` is unusable and cannot be fixed.** Every kept row is `high`.

**The 596 rejected rows are still labelled at v2.** This is deliberate: no codebook
revision after v2 touched the gate rules. Revisit only if a gate rule changes.

## Decisions that could be revisited

- **The labelling cutoff is launch (2021-12-25), not first light.** The old NEXT.md
  proposed moving it to first light (2022-07-12) to save about $0.40. The kept data
  says no: 10 observations are dated between the two.
- **The Streamlit explorer and the ResNet/XGBoost classifiers were removed.** The
  classifiers were trained on the tag labels this project replaced. Both are in git
  history at commit 0af5a0a if wanted back. Retraining on the golden labels would be
  the honest version of that work.
- **Downscaled dataset images and embeddings are not committed.** They are
  regenerated by the pipeline.

## Possible extensions

Joining to real science data through `object_name_normalized` is the obvious next
step. MAST needs no API key:

```python
from astroquery.mast import Observations
Observations.query_object("NGC 3132", radius="0.02 deg")
```

Note that M51 is also NGC 5194, so unifying designations across catalogues needs a
cross-reference lookup that the normaliser does not attempt.

If the dataset is ever relabelled wholesale, the Batch API halves the cost.

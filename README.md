# JWST Golden Dataset

An Airflow pipeline that ingests the NASA Webb Telescope's Flickr archive and turns it
into a labelled dataset of real astronomical observations, separated from the far
larger volume of hardware photos, press events and community artwork published
alongside them.

**The dataset:** 481 observations labelled by subject, modality, object and instrument,
plus the 596 photos rejected along the way, each with a reason. It lives in
[`include/data/dataset/`](include/data/dataset).

**The pipeline:** four DAGs covering incremental ingest, image embeddings, LLM
labelling and a write-audit-publish dataset build, with data quality gates between
stages and a test suite that runs without a scheduler.

---

## Why this exists

The NASA Webb Flickr account holds over 4,000 photos, and most of them are not
astronomy. Roughly a third are mirror assembly, clean rooms, launch coverage and
conference photos from the decade before launch, and 247 more are
`#UnfoldTheUniverse` community artwork.

Labelling them by their Flickr tags does not work. Flickr strips the spaces from
tags, so a rule table matching `star cluster` or `black hole` never fires. An earlier
version of this project did exactly that, with these results:

- Globular clusters were filed as `star`.
- Planetary nebulae and protoplanetary disks were filed as `exoplanet`, because a bare
  `planet` fragment matched `planetarynebula`.
- No photo ever received the `black_hole` label.
- About 93 real observations were filed as `observatory / engineering`, the catch-all
  that fires when a photo's only tags are `jwst`, `webb`, `space` and `telescope`.

Labelling from the caption and the image instead removes that entire class of error.

## Pipeline

```
                    ┌─────────────┐  photos   ┌────────────┐  embeddings
  Flickr API ──────▶│ jwst_ingest │──────────▶│ jwst_embed │─────────────┐
     @daily         └─────────────┘           └────────────┘             ▼
                                                                  ┌──────────────┐
                    ┌─────────────┐  label log                    │ jwst_dataset │──▶ dataset/
  Claude API ──────▶│ jwst_label  │──────────────────────────────▶│ write·audit· │
     manual         └─────────────┘                               │   publish    │
                                                                  └──────────────┘
                         all state in DuckDB + include/data/
```

| DAG | Trigger | What it does |
|---|---|---|
| `jwst_ingest` | daily | Lists every photo on the account, diffs the ids against the warehouse, fetches captions and downloads originals |
| `jwst_embed` | new photos | ResNet50 embeddings, computed in parallel mapped tasks and loaded by a single writer |
| `jwst_label` | manual | Labels unlabelled photos with Claude, capped per run by `max_photos` |
| `jwst_dataset` | new labels or embeddings | Rebuilds the dataset in staging, audits it, and publishes it only if every check passes |

The DAGs trigger one another through Airflow assets rather than on clocks. A stage
only announces new data when something actually changed, so a quiet day on Flickr
ends at the ingest.

## Engineering decisions

**Every stage is incremental and driven by warehouse state.** The ingest compares ids
rather than dates, because Flickr's `date_taken` is supplied by the uploader: the
Horsehead Nebula is dated 2124. Downloads, embeddings and labels are each planned from
what the warehouse lacks, not from what the previous task passed along. A failure
today is therefore retried tomorrow without intervention.

**Writing to DuckDB is serialised deliberately.** DuckDB lets a single process hold
the file for writing, and that writer locks out readers in other processes as well.
Every task that opens the warehouse runs in a one-slot Airflow pool, and network calls
are split into separate tasks, so no task holds the pool while waiting on Flickr. A
test inspects every task's source to enforce the pool rule.

**Downloads are atomic and verified.** The original ingest streamed each download to
its final path and skipped any photo whose file already existed. Nine downloads that
were cut off part-way, one of them empty, sat in the archive for months failing to
decode, and nothing would ever fetch them again. Downloads now go to a `.part` file,
are checked against `Content-Length`, are decoded in full, and only then are renamed
into place. The migration found all nine and queued them for download again.

**The dataset is published with write-audit-publish.** Each build is written to
staging and checked there:

- keys are unique
- values come from the codebook
- each release group has exactly one primary image
- kept plus rejected rows account for every label
- the row count has not dropped below the last published build

Only a staging build that passes every check replaces the published files. The
manifest carries checksums and no timestamps, so an unchanged rebuild is
byte-identical and produces no git diff.

**Labelling is the one manual stage, because it costs money.** Every response is
appended to a label log the moment it arrives, and that log is committed to git. The
warehouse `labels` table is rebuilt from it. This has three consequences:

- An interrupted run never pays twice for the same photo.
- A fresh clone rebuilds the whole dataset without paying for labelling again.
- Loading the log rejects any value outside the codebook. An early run put a modality
  value into `gate_category`, and five real observations, the Horsehead Nebula among
  them, were silently counted as rejections.

**Secrets stay out of logs.** Flickr takes its API key as a query parameter, so the
exception messages from `requests` contain it, and Airflow writes full tracebacks,
chained exceptions included, into task logs. The Flickr client rebuilds every error
with the key redacted and the original exception unchained, and a test asserts it.

**Airflow is kept at the edges.** The DAG files only decide ordering, retries and
concurrency. All logic lives in `include/jwst_pipeline/`, which never imports Airflow,
so the tests exercise it directly against temporary warehouses.

## Running it

**Prerequisites:** [Astro CLI](https://www.astronomer.io/docs/astro/cli/install-cli),
Docker Desktop, and a [Flickr API key](https://www.flickr.com/services/apps/create/).
Labelling also needs an Anthropic API key.

```bash
cp .env.example .env            # add FLICKR_API_KEY, and ANTHROPIC_API_KEY to label
astro dev start                 # Airflow UI at http://localhost:8080
```

New DAGs start paused. Unpause `jwst_ingest`, `jwst_embed` and `jwst_dataset` to run
the automatic part of the pipeline. Then trigger `jwst_label` whenever there is
something to label:

```bash
astro dev run dags trigger jwst_label --conf '{"max_photos": 50}'
```

On a fresh clone, the first `jwst_dataset` run loads the committed label log, so the
published dataset is rebuilt from Flickr without any labelling spend.

Tests run on the host or in the containers:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt \
    --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.1.6/constraints-3.10.txt
.venv/bin/pytest

astro dev pytest                # includes the torch embedding test
```

## The dataset

`include/data/dataset/jwst_space_images.parquet`, with a matching CSV, holds one row
per observation.

| Field | Description |
|---|---|
| `photo_id` | Flickr id, primary key |
| `title`, `description` | Original NASA/ESA caption text |
| `subject` | One of 13 classes: `galaxy`, `galaxy_cluster`, `deep_field`, `nebula`, `planetary_nebula`, `supernova_remnant`, `star`, `star_cluster`, `protoplanetary_disk`, `exoplanet`, `solar_system`, `black_hole`, `other_astronomical` |
| `modality` | `image`, `annotated_image`, `spectrum_or_plot` or `comparison_composite` |
| `object_name` | Catalogue designation copied verbatim from the caption |
| `object_name_normalized` | The same name in canonical form, so `M51` and `Messier 51` group together |
| `object_name_verified` | Whether the name was found verbatim in the caption |
| `instrument` | NIRCam, MIRI, NIRSpec, NIRISS, FGS, multiple, non-Webb or unknown |
| `release_group` | Title with its parenthetical qualifier removed, grouping renderings of one figure |
| `near_duplicate_group` | Cluster of images whose embeddings exceed 0.95 cosine similarity |
| `is_primary` | One representative per release group |
| `rationale`, `confidence` | The model's reasoning and self-reported certainty |
| `flickr_url`, `date_taken`, `flickr_tags` | Original metadata |
| `source_tag_label` | The old keyword-derived label, kept for comparison; null for photos ingested since it was retired |
| `image_file` | Path of the 1024px copy, relative to the dataset directory |
| `has_embedding`, `model`, `codebook_version`, `labelled_with_vision` | Provenance |

`rejected.parquet` holds every photo filtered out, with the reason. `manifest.json`
records row counts, class counts and checksums. The downscaled images are regenerated
by the pipeline and are not committed.

## How the labels were made

Each candidate photo is sent to Claude Haiku 4.5 with its title, caption and image,
and the label definitions from [`codebook.md`](codebook.md) as the system prompt. The
model first decides whether the photo is an observation at all, then assigns subject,
modality, object name and instrument. Strict tool use constrains the response to the
schema.

Sending the image matters more than it appears. One photo of Sagittarius A* has a
caption describing 48 hours of genuine Webb observing time, but the image itself
carries the words "Artist's Concept" in the corner. No amount of reading the text
would catch that.

Candidates are photos dated on or after launch, or with no date. Photos dated before
launch are skipped, as they were when the dataset was first built. That build also
labelled the pre-launch photos the old tag matcher had reached, and two of those
turned out to be observations with wrong dates, so a real observation misdated into
the past can still be missed. Labelling the original 1,077 candidates cost roughly
five dollars; new photos cost about half a cent each.

## Known limitations

- **Nobody has checked the labels.** They are model-generated and no accuracy figure
  exists. `jwst_dataset` writes a review sheet to `include/data/review/review.html`
  that samples rows across subject classes for hand-checking.
- **`confidence` carries no information.** Every kept row came back `high`.
- **`instrument` is the weakest field.** The model reaches for `multiple` on
  multi-observatory releases even when the caption names no Webb instrument.
- **`subject` is single-label**, although many images genuinely contain several kinds
  of object. The label reflects what the caption presents as the subject.
- **`date_taken` is passed through unmodified** and is unreliable.
- **These are press images, not science data.** They are colour composites stretched,
  cropped and cleaned for publication, several irreversible steps removed from the
  FITS files in [MAST](https://mast.stsci.edu). They suit classification and
  similarity work; they cannot be used for photometry or astrometry.

## Layout

```
dags/                       orchestration only: order, retries, pools, assets
include/jwst_pipeline/
  flickr.py                 REST client: retries, pacing, key redaction
  images.py                 atomic downloads, decode verification, downscaling
  ingest.py                 ingest steps
  embeddings.py             ResNet50 vectors, staged then loaded
  codebook.py               enums, schema and the prompt the model sees
  labelling.py              model calls, the label log, the queue
  assemble.py               build, audit and publish the dataset
  quality.py                data quality checks
  warehouse.py              DuckDB connections and migrations
  review.py                 HTML sheet for hand-checking labels
include/data/
  dataset/                  the published dataset          (tracked)
  labels/raw_labels.jsonl   every model response           (tracked)
  warehouse/, images/       DuckDB file and originals      (local, 18 GB)
scripts/                    one-off migration, label-run comparison
tests/                      unit, integration and DAG structure tests
codebook.md                 the labelling specification
```

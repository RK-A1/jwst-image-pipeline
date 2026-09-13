# JWST Golden Dataset

A curated set of 481 James Webb Space Telescope observations, labelled by subject,
separated from the much larger archive of hardware photos, press events and community
artwork that NASA publishes alongside them.

## What this is

The NASA Webb Flickr account holds 4,345 photos, but most of them are not
astronomy. Roughly a third are mirror assembly, clean rooms, launch coverage and
conference photos from the decade before the telescope launched, and a further 247 are
`#UnfoldTheUniverse` community artwork. This dataset is the subset that consists of
actual observations, with labels describing what each one shows.

It exists because labelling these images by their Flickr tags does not work. Flickr
returns tags without spaces, so a rule table matching phrases like `star cluster` or
`black hole` never fires. In the source project this meant globular clusters were
filed as `star`, planetary nebulae and protoplanetary disks were filed as `exoplanet`
because a bare `planet` fragment matched `planetarynebula`, and no photo ever received
the `black_hole` label at all. Around 93 real observations were filed as
`observatory / engineering`, the catch-all that fires when a photo's only tags are
`jwst`, `webb`, `space` and `telescope`.

Labelling from the caption and the image instead removes that entire class of error.

## Where the data comes from

The photos and metadata were collected by [RK-A1/JWST](https://github.com/RK-A1/JWST),
an Airflow pipeline that paginates the Flickr API, downloads new photos, stores their
metadata in DuckDB and extracts ResNet50 embeddings. That project owns ingestion. This
one read its database once, read-only, and never wrote back to it.

The source archive is frozen. Its ingest requires a Flickr API key that is no longer
available, so the 4,345 photos will not grow, and this dataset covers all of them up
to 6 May 2026.

## Schema

`data/jwst_space_images.parquet` and the matching CSV hold one row per observation.

| Field | Description |
|---|---|
| `photo_id` | Flickr ID, primary key |
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
| `source_tag_label` | The old keyword-derived label, kept for comparison |

`data/rejected.parquet` holds the 596 photos that were filtered out, each with the
reason. `data/embeddings.parquet` holds the 2048-dimensional ResNet50 vectors carried
over from the source project. `images/` holds copies downscaled to 1024px.

## How it was built

Candidates are the union of two signals: photos the old tag labeller had reached, and
photos dated on or after launch. Neither is reliable alone, because Flickr's
`date_taken` is uploader-supplied and sometimes wrong. The Horsehead Nebula and
Jupiter's Great Red Spot both carry dates in 2124, and a date filter alone discards
them. The union is 1,077 photos.

Each was sent to Claude Haiku 4.5 with its title, caption and image, together with the
label definitions in [`codebook.md`](codebook.md). The model first decides whether the
photo is an observation at all, then assigns the subject, modality, object name and
instrument. Responses are constrained to the schema by strict tool use.

Sending the image matters more than it appears. One photo of Sagittarius A* has a
caption describing 48 hours of genuine Webb observing time, but the image itself
carries the words "Artist's Concept" in the corner. No amount of reading the text
would catch that.

```bash
python build/00_extract.py                                            # read source DB
python build/01_label.py --backend api --model claude-haiku-4-5 --vision
python build/02_assemble.py --images                                  # build outputs
python build/03_review.py --n 100                                     # review sheet
```

Labelling costs roughly five dollars and takes about forty minutes. The scripts also
support local inference through Ollama with `--backend ollama`, which is free but
substantially slower and less accurate.

## Known limitations

Labels are model-generated and have not been verified by an expert, so treat them as
good rather than authoritative. `build/03_review.py` produces a review sheet for
hand-checking a stratified sample, and no accuracy figure is claimed here until that
has been done.

The `confidence` field carries no information. Every kept row came back `high`, with
no `medium` or `low` at all, so it cannot be used to find uncertain rows.

The `instrument` field is the weakest. Of the 156 rows marked `multiple`, 101 name no
Webb instrument anywhere in their caption and should read `unknown` instead. The
value is legible enough to filter on, but do not trust it.

`subject` is single-label, and many images genuinely contain several kinds of object.
The label reflects what the caption presents as the subject. Where a caption is vague
or wrong, the label inherits that.

`date_taken` is passed through unmodified and is unreliable. `flickr_url` points at
the Flickr page rather than the image file, because the source database never stored
the direct URLs.

Finally, these are press images rather than science data. They are colour composites
that have been stretched, cropped and cleaned for publication, several irreversible
steps removed from the FITS files in
[MAST](https://mast.stsci.edu). They suit image
classification and similarity work; they cannot be used for photometry or astrometry.

## Layout

```
codebook.md              label definitions, and the tie-breakers that decide hard cases
build/
  00_extract.py          read the source DuckDB
  01_label.py            label via Claude API or local Ollama
  02_assemble.py         join, validate, group, write outputs
  03_review.py           build an HTML sheet for hand-checking
  compare_versions.py    diff two labelling runs field by field
  run_overnight.sh       run unattended, guarding against sleep and battery loss
data/                    the dataset, plus every raw model response
images/                  downscaled copies of the kept images
```

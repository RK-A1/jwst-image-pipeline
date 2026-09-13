# Where this left off

Last worked on 13 September 2026. The dataset is built, committed and pushed to
`RK-A1/jwst-golden` (private). Everything below is what remains.

## State

481 observations labelled at codebook v6, out of 1,077 candidates drawn from the 4,345
photos in [RK-A1/JWST](https://github.com/RK-A1/JWST). All 13 subject classes have
members. Total API spend was about $8.20; roughly $1.80 of the original $10 credit
remains, so anything costing more than that needs a top-up first.

The source project was never modified. Its DuckDB was read once, read-only.

## The one thing that actually matters

**Nobody has checked the labels.** They are model-generated and no accuracy figure
exists. `build/03_review.py` builds an HTML sheet that samples 100 rows stratified
across subject classes, shows each image beside its label, rationale and caption, and
lets you tick anything wrong and copy out the corrections as JSON.

The sheet currently on disk was generated against the older v2 labels, so regenerate
it before using it:

```bash
python build/03_review.py --n 100
open review/review.html
```

Two things to look at particularly: the 13 rows where `object_name_verified` is false,
and the `galaxy` / `nebula` / `deep_field` boundaries, which is where the codebook
does the most work and where the labels are least certain. When the correction rate is
known, put it in the README's limitations section, which currently claims no accuracy
figure at all.

## Known defects, in priority order

**`instrument = multiple` is wrong on about 101 rows.** Codebook v6 improved this from
181 wrong to 101, but the model still reaches for `multiple` on multi-observatory
releases even when the caption names no Webb instrument. The rule says `unknown` is
correct when nothing is named.

This does not need another labelling pass. Every one of those rows can be corrected
deterministically, because the caption text is already in the dataset: if
`instrument = 'multiple'` and the title and description together name fewer than two
of NIRCam, MIRI, NIRSpec, NIRISS or FGS, the value should be the single named
instrument, or `unknown` if none is named. That is a post-processing step in
`02_assemble.py` and costs nothing. It is the highest-value cheap fix left.

**`object_name` drifted slightly at v6.** A few rows now expand or substitute names
against the codebook's own instruction to copy verbatim: `30 Doradus` became
`Tarantula Nebula`, and `WR 140` became `Wolf-Rayet 140`. Minor, and
`object_name_normalized` absorbs some of it, but the same deterministic approach would
work — flag any `object_name` absent from the caption and fall back to the v2 value in
`data/raw_labels.v2.bak.jsonl`.

**`confidence` is unusable and cannot be fixed.** Every kept row is `high`. It stays in
the schema because it costs nothing, but it will never identify uncertain rows on this
corpus. Ignore it.

**The 596 rejected rows are still labelled at v2.** This is deliberate and almost
certainly fine, because no codebook revision after v2 touched the gate rules — v3
through v6 only changed modality and instrument, which are `not_applicable` and
`unknown` for rejects. Re-labelling them would cost about $2.60 to change nothing. Only
revisit if a future revision changes a gate rule.

## Before any future re-run

**Move the date cutoff in `build/00_extract.py` from `2021-12-25` to `2022-07-12`.**
The current value is launch day; first light was seven months later. That mistake put
92 launch-coverage photos into the candidate set which had no chance of passing the
gate, costing about $0.40. Changing it drops those 92 and keeps every tag-labelled
photo, so nothing real is lost. I left the current value in place so the committed
dataset matches the code that produced it.

**Keep `codebook.md` and the prompt in `build/01_label.py` in sync.** They duplicate
the same rules in two files and there is no mechanism enforcing agreement. This is the
main structural hazard in the project. Every labelled row carries `codebook_version`,
so drift is at least detectable after the fact.

**Leave `strict: true` on the tool definition.** Without it the schema enums are
advisory, and the model will occasionally emit an out-of-enum value. That happened on
the first run: five records came back with a modality value in `gate_category`, and
because anything other than `astronomical_observation` counts as a rejection, five
genuine observations — the Horsehead Nebula among them — were silently dropped rather
than raising an error. `02_assemble.py` now checks for this and says so loudly.

**Use `build/compare_versions.py` after any codebook change.** It diffs two label runs
field by field. Judge a revision by what it actually changed, not by whether it sounded
like an improvement; v3 looked complete and left two larger defects untouched.

## Possible extensions

Joining to real science data through `object_name_normalized` is the obvious next step.
MAST needs no API key, unlike Flickr:

```python
from astroquery.mast import Observations
Observations.query_object("NGC 3132", radius="0.02 deg")
```

Note that M51 is also NGC 5194, and unifying designations across catalogues needs a
cross-reference lookup that the normaliser does not attempt.

If the dataset is ever rebuilt repeatedly, the Batch API halves the cost, at the price
of writing and debugging a polling loop. At around $5 a run that was not worth it once.
It would be worth it at five runs.

## Resuming

```bash
cd /Users/rk/ds/jwst-golden
export ANTHROPIC_API_KEY=...          # or put it in .env, which is gitignored

python build/01_label.py --backend api --model claude-haiku-4-5 --vision
python build/02_assemble.py --images
python build/03_review.py --n 100
```

Labelling appends to `data/raw_labels.jsonl` and skips any `photo_id` already present,
so it is safe to interrupt and rerun; it picks up where it stopped and re-bills
nothing. To force a re-label, delete those rows from the JSONL first and rerun.

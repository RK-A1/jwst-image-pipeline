# Codebook

Label definitions for the JWST golden dataset. This file is the specification the
model is prompted with — if a label is wrong, the fix usually belongs here, not in
the code.

Every photo gets five fields: a **gate** decision, and — if it passes — a
**subject**, a **modality**, an **object name**, and an **instrument**.

---

## 1. Gate — `gate_category`

The question: *is this actual observational data of something in space?*

| Value | Keep? | Definition |
|---|---|---|
| `astronomical_observation` | **yes** | Data captured by a telescope of an object outside Earth's atmosphere. Includes images, mosaics, spectra, light curves, and multi-observatory composites. |
| `hardware_engineering` | no | The telescope or its components: mirrors, sunshield, instruments, cryo chambers, clean rooms, testing rigs, launch vehicles, ground stations. |
| `people_event` | no | Humans as the subject: ceremonies, press conferences, briefings, museum exhibits, staff portraits, outreach events. |
| `artwork_illustration` | no | Artist's concepts, renderings, `#JWSTArt` community submissions, infographics with no observational data. |
| `promotional_other` | no | Logos, posters, anniversary graphics, mission patches, social cards, broadcast stills, anything else. |

### Gate tie-breakers

- **Data rendered as a chart still counts.** A transmission spectrum of an exoplanet
  atmosphere is `astronomical_observation`. The axes do not make it a graphic.
- **An artist's concept never counts,** even when it depicts a real object Webb
  observed. The test is whether photons from the object produced the picture.
- **Text printed in the image overrides the caption.** NASA marks renderings with
  "Artist's Concept", "Illustration", "Simulation", or "Artist's Impression", usually
  small and in a corner. When that text is visible the photo is
  `artwork_illustration`, however much the caption discusses real observations — and
  the caption usually does, because the release is about real science. This is the
  single case where the caption is actively misleading and only the image is
  reliable.
- **A composite with non-Webb data counts** — Chandra, Hubble, Euclid, Spitzer,
  ground-based. It is still observational data.
- **A telescope image embedded in a promotional layout** counts if the observation
  is the substance of the image; it does not if the observation is a small
  decorative element in a poster.
- **When the caption describes an event but shows an observation** (e.g. "First
  Images Broadcast"), judge the image, not the occasion.

---

## 2. Subject — `subject`

The primary astronomical object. Assign exactly one. Where an image contains
several kinds of object, choose what the caption presents as the subject.

| Value | Definition |
|---|---|
| `galaxy` | One galaxy, or a small interacting/merging group. Includes spirals, ellipticals, irregulars, starburst galaxies. |
| `galaxy_cluster` | A gravitationally bound cluster. Includes lensing clusters (Abell, SMACS, El Gordo, Bullet Cluster). |
| `deep_field` | A wide survey field whose subject is the population of background galaxies rather than any single object (JADES, CEERS, COSMOS-Web, Ultra Deep Field). |
| `nebula` | Interstellar gas and dust: emission, reflection, and dark nebulae; HII regions; star-forming regions; molecular clouds. |
| `planetary_nebula` | The ejected envelope of a dying low-mass star (Ring, Southern Ring, NGC 3132). **Not** related to planets. |
| `supernova_remnant` | The expanding debris of an exploded star (Cassiopeia A, Crab, SN 1987A). |
| `star` | An individual star or stellar system: binaries, brown dwarfs, white dwarfs, Wolf-Rayet stars, variable stars. |
| `star_cluster` | A globular or open cluster of stars. |
| `protoplanetary_disk` | Circumstellar and debris disks, protostars, planet-forming regions, Herbig-Haro objects and protostellar jets. |
| `exoplanet` | A planet outside the solar system, or its atmosphere. Includes transit and direct-imaging studies. |
| `solar_system` | Objects within our solar system: planets, moons, rings, asteroids, comets, Kuiper Belt objects. |
| `black_hole` | Black holes and their immediate environment: AGN, quasars, Sgr A*, relativistic jets. |
| `other_astronomical` | A real observation that fits none of the above. Use sparingly. |

### Subject tie-breakers

These are the distinctions the previous tag-based system got wrong. They matter most.

- **`planetary_nebula` is never `exoplanet`.** The word "planetary" is historical and
  describes the round telescopic appearance, not planets.
- **`protoplanetary_disk` is not `exoplanet`.** A disk where planets may form is a
  disk. Only assign `exoplanet` when a planet itself is the subject.
- **`supernova_remnant` outranks `nebula`.** Both are glowing gas; the remnant of an
  explosion is the more specific label.
- **`planetary_nebula` outranks `nebula`** for the same reason.
- **`deep_field` vs `galaxy_cluster` — decide by what the caption is *about*, not by
  what is named.** Use `deep_field` when the subject is the population of distant or
  background galaxies, the depth of the exposure, or the early universe — even when a
  foreground lensing cluster is named in the text. Use `galaxy_cluster` when the
  subject is the cluster itself: its mass, structure, lensing properties, or its
  member galaxies. "Deepest infrared image of the universe yet" is `deep_field` even
  though it is an image of SMACS 0723; "Webb pierces the Bullet Cluster, refines its
  mass" is `galaxy_cluster`.
- **A patch inside a galaxy is `nebula`, not `galaxy`,** when the subject is the
  interstellar material — dust lanes, star fields, HII regions — even if the host
  galaxy is named. Use `galaxy` only when the galaxy as a whole is the subject: its
  morphology, structure, or identity as a system. A MIRI test frame showing dust and
  stars in part of the Large Magellanic Cloud is `nebula`.
- **Sgr A* is `black_hole`, not `star`,** despite the name.
- **A star-forming region is `nebula`,** not `star`, unless one protostar or disk is
  the subject (then `protoplanetary_disk`).
- **A galaxy hosting an AGN is `black_hole`** only if the caption is about the AGN,
  jet, or central engine. Otherwise `galaxy`.
- **Comets and asteroids are `solar_system`,** never `star`.

---

## 3. Modality — `modality`

How the data is presented. Independent of subject.

| Value | Definition |
|---|---|
| `image` | A direct image or mosaic, unannotated. |
| `annotated_image` | An image with graphics added on top of the observation. Counts as annotated: text or object labels; arrows, crosshairs, tick marks; circles, ellipses, dashed field-of-view boxes; **contour lines**; region markers; inset boxes and the callout lines joining them; compass roses; scale bars; colour/filter keys ("blue = F090W"); colour bars; coordinate grids; panel letters; a **star glyph or occulting disc marking a masked host star** in coronagraphic exoplanet images; a text strip added above or below the frame. If a graphic was drawn over the pixels it is annotated, even when the marks carry no text. |
| `spectrum_or_plot` | Data plotted on axes: spectra, light curves, transmission curves. |
| `comparison_composite` | **Two or more separate panels** presented together in one graphic — Webb vs. Hubble side by side, NIRCam beside MIRI, before/after, a strip of insets with their own frames. The test is whether you can see distinct panel boundaries. A single picture that blends data from several telescopes into one frame is **not** this — "NASA's Chandra Adds X-ray Vision to Webb Images" is one merged picture, so it is `image`. NASA uses "composite" to mean blended; here it means panelled. |

If an image is both annotated and a comparison, prefer `comparison_composite`.
**Modality is decided by the image, not the caption.** Count the panels you can
actually see. Ignore the words "composite", "combined", "merged", "blended" and
"multi-observatory" in the text — those describe how the data was processed, not how
the graphic is laid out. Webb + Chandra data merged into one frame is one frame:
`image`. Only visible panel divisions make it `comparison_composite`.


**Not annotation — these are the data, not overlays:**

- Diffraction spikes, the six-pointed star pattern from Webb's mirror segments.
- Ragged or stepped image edges, which are the detector mosaic footprint.
- Black background, letterboxing, or a non-square crop.
- False colour itself. Every one of these images is false colour; that is processing,
  not annotation.
- Noise, detector artifacts, or cosmic-ray hits.

Clean and annotated versions of the same frame are frequently published as a pair,
and the only difference may be a few drawn lines. Look closely before choosing
`image`.


---

## 4. Object name — `object_name`

The catalogue designation or proper name of the primary object, copied **verbatim**
from the title or caption. Examples: `NGC 6334`, `WASP-96 b`, `SMACS 0723`,
`Cassiopeia A`, `Pillars of Creation`.

- Copy exactly as written. Do not normalise, expand, or correct it.
- Prefer the catalogue designation when both appear (`NGC 3132` over `Southern Ring Nebula`).
- Use `null` when no specific object is named. Do not invent one.

The build validates this field: any `object_name` that does not appear verbatim in the
title or description is flagged in `object_name_verified`. Small models mangle
catalogue numbers, and this is the field where that does the most damage.

---

## 5. Instrument — `instrument`

One of: `NIRCam`, `MIRI`, `NIRSpec`, `NIRISS`, `FGS`, `multiple`, `non-Webb`, `unknown`.

- **`unknown` is the correct answer whenever the text does not name an instrument.**
  It is not a failure or a hedge — most captions simply never say. Do not infer an
  instrument from the subject, from how the image looks, or from the kind of
  observation.
- **`multiple` requires two or more Webb instruments to be named explicitly.** One
  named instrument means that instrument, not `multiple`.
- The instrument is most often in the title, in parentheses — `(NIRCam Image)`,
  `(NIRSpec MSA Emission Spectra)`, `(NIRCam and MIRI)`.
- `non-Webb` when the data is entirely from another observatory.
- For a Webb + Chandra composite, name the Webb instrument if stated, else `unknown`.

---

## 6. Confidence — `confidence`

`high` / `medium` / `low`. Use `low` when the caption is thin or the classification is
genuinely contested, not merely when the object is unfamiliar. Rows marked `low` are
the review queue.

---

## Revision log

- **v2** (2026-09-12) — three rules added after a 16-record benchmark across
  `gemma4:e2b`, Claude Sonnet 5, and Claude Haiku 4.5. (1) Text printed in the image
  overrides the caption: an image of Sagittarius A* whose caption describes 48 hours
  of real Webb observing carries "Artist's Concept" in the corner, and all three
  models correctly rejected it only because they could see the pixels. (2) A
  `deep_field` / `galaxy_cluster` tie-breaker based on what the caption is about
  rather than what it names — the one case Sonnet got wrong. (3) A patch inside a
  galaxy showing interstellar material is `nebula`, not `galaxy`.
  **The prompt in `build/01_label.py` duplicates these rules and must be kept in
  sync with this file; `codebook_version` is stamped on every labelled row.**

- **v1** (2026-09-12) — initial. Taxonomy derived from failure analysis of the
  `TAG_RULES` substring matcher in the source project, which collapsed `star_cluster`
  into `star`, labelled planetary nebulae and protoplanetary disks as `exoplanet`, and
  produced zero `black_hole` rows because every one of its multi-word keyword
  fragments was unmatchable against space-free Flickr tags.

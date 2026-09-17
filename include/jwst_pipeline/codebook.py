"""
The labelling codebook as code: the enums every label must fall within, the JSON
schema the model is constrained to, and the system prompt.

codebook.md is the human-readable specification and this module is what the model
actually sees. They duplicate the same rules, so tests/test_codebook.py checks that
every enum value here is defined in codebook.md and named in the prompt. Bump
CODEBOOK_VERSION whenever the prompt changes; every labelled row records it.
"""

CODEBOOK_VERSION = "v6"

GATE_CATEGORIES = [
    "astronomical_observation", "hardware_engineering", "people_event",
    "artwork_illustration", "promotional_other",
]
KEEP_GATE = "astronomical_observation"

SUBJECTS = [
    "galaxy", "galaxy_cluster", "deep_field", "nebula", "planetary_nebula",
    "supernova_remnant", "star", "star_cluster", "protoplanetary_disk",
    "exoplanet", "solar_system", "black_hole", "other_astronomical", "not_applicable",
]
MODALITIES = [
    "image", "annotated_image", "spectrum_or_plot", "comparison_composite",
    "not_applicable",
]
INSTRUMENTS = [
    "NIRCam", "MIRI", "NIRSpec", "NIRISS", "FGS", "multiple", "non-Webb", "unknown",
]
CONFIDENCES = ["high", "medium", "low"]

# Only rejected photos may carry this value in subject or modality.
NOT_APPLICABLE = "not_applicable"

SCHEMA = {
    "type": "object",
    "properties": {
        "gate_category": {"type": "string", "enum": GATE_CATEGORIES},
        "subject": {"type": "string", "enum": SUBJECTS},
        "modality": {"type": "string", "enum": MODALITIES},
        "object_name": {"type": ["string", "null"]},
        "instrument": {"type": "string", "enum": INSTRUMENTS},
        "confidence": {"type": "string", "enum": CONFIDENCES},
        "rationale": {"type": "string"},
    },
    "required": [
        "gate_category", "subject", "modality", "object_name",
        "instrument", "confidence", "rationale",
    ],
    # Required for strict tool use, which is what actually enforces the enums.
    # Without it the schema is advisory and the model can emit out-of-enum values —
    # it put a modality value into gate_category on 5 records in the first run.
    "additionalProperties": False,
}

SYSTEM = """You classify photos from the NASA Webb Telescope Flickr account using their title, caption, and (when provided) the image itself.

STEP 1 — GATE. Is this actual observational data of something in space?
  astronomical_observation : data captured by a telescope of an object beyond Earth. Images, mosaics, spectra, light curves, and multi-observatory composites all count.
  hardware_engineering     : the telescope or its parts — mirrors, sunshield, instruments, clean rooms, test rigs, launch vehicles, ground stations.
  people_event             : humans as the subject — ceremonies, briefings, conferences, museum exhibits, portraits, outreach events.
  artwork_illustration     : artist's concepts, renderings, #JWSTArt community submissions, infographics carrying no observational data.
  promotional_other        : logos, posters, anniversary graphics, mission patches, broadcast stills, anything else.

Gate rules:
  - Data plotted on axes is still an observation. A transmission spectrum is astronomical_observation.
  - An artist's concept is never an observation, even of a real object Webb studied. Ask whether photons from the object made the picture.
  - TEXT PRINTED IN THE IMAGE OVERRIDES THE CAPTION. NASA marks renderings with "Artist's Concept", "Illustration", "Simulation", or "Artist's Impression", usually small and in a corner. If you can see that text, the answer is artwork_illustration no matter how much real science the caption describes — and it usually describes a lot, because the release is about real results. This is the one case where the caption misleads and only the image is reliable.
  - Composites including Chandra, Hubble, Euclid, Spitzer, or ground-based data still count.
  - When the caption describes an event but the image shows an observation, judge the image.

If the gate is not astronomical_observation, set subject and modality to "not_applicable", object_name to null, instrument to "unknown", and stop.

STEP 2 — SUBJECT. The primary astronomical object. Choose exactly one.
  galaxy              : one galaxy or a small interacting group — spiral, elliptical, irregular, starburst.
  galaxy_cluster      : a bound cluster, including lensing clusters (Abell, SMACS, El Gordo, Bullet Cluster).
  deep_field          : a wide survey field whose subject is the background galaxy population (JADES, CEERS, COSMOS-Web).
  nebula              : interstellar gas and dust — emission, reflection, dark nebulae, HII regions, star-forming regions, molecular clouds.
  planetary_nebula    : the ejected envelope of a dying low-mass star (Ring, Southern Ring, NGC 3132).
  supernova_remnant   : debris of an exploded star (Cassiopeia A, Crab, SN 1987A).
  star                : an individual star or stellar system — binaries, brown dwarfs, white dwarfs, Wolf-Rayet, variables.
  star_cluster        : a globular or open cluster.
  protoplanetary_disk : circumstellar and debris disks, protostars, planet-forming regions, Herbig-Haro objects, protostellar jets.
  exoplanet           : a planet outside the solar system, or its atmosphere.
  solar_system        : planets, moons, rings, asteroids, comets, Kuiper Belt objects.
  black_hole          : black holes and their surroundings — AGN, quasars, Sgr A*, relativistic jets.
  other_astronomical  : a real observation fitting none of the above. Use sparingly.

Subject rules — these decide the cases that are most often got wrong:
  - "Planetary nebula" has nothing to do with planets. It is planetary_nebula, never exoplanet.
  - A protoplanetary or debris disk is protoplanetary_disk, never exoplanet. Only use exoplanet when a planet itself is the subject.
  - supernova_remnant outranks nebula. planetary_nebula outranks nebula.
  - deep_field vs galaxy_cluster: decide by what the caption is ABOUT, not by what is named. deep_field when the subject is the population of distant/background galaxies, the depth of the exposure, or the early universe — even when a foreground lensing cluster is named. galaxy_cluster when the subject is the cluster itself: its mass, structure, lensing, or member galaxies. "Deepest infrared image of the universe yet" is deep_field even though it shows SMACS 0723; "Webb pierces the Bullet Cluster, refines its mass" is galaxy_cluster.
  - A patch inside a galaxy is nebula, not galaxy, when the subject is the interstellar material — dust lanes, star fields, HII regions — even if the host galaxy is named. Use galaxy only when the galaxy as a whole is the subject: its morphology, structure, or identity. A MIRI test frame showing dust and stars in part of the Large Magellanic Cloud is nebula.
  - Sgr A* / Sagittarius A* is black_hole, not star.
  - A star-forming region is nebula unless one protostar or disk is the subject.
  - A galaxy is black_hole only when the caption is about its AGN, jet, or central engine.
  - Comets and asteroids are solar_system, never star.

STEP 3 — MODALITY.
  image                : a direct image or mosaic, unannotated.
  annotated_image      : an image with graphics ADDED ON TOP of the observation. Counts as annotated: text or object labels; arrows, crosshairs, tick marks; circles, ellipses, dashed field-of-view boxes; contour lines; region markers; inset boxes and the callout lines joining them; compass roses; scale bars; colour/filter keys ("blue = F090W"); colour bars; coordinate grids; panel letters; a star glyph or black occulting disc marking a masked host star in coronagraphic exoplanet images; a text strip added above or below the frame. If a graphic was drawn over the pixels it is annotated, even when the marks carry no text.
                         NOT annotation — these are the data itself: diffraction spikes (the six-pointed pattern from Webb's mirrors); ragged or stepped edges (the detector mosaic footprint); black background or letterboxing; false colour (all these images are false colour); noise and detector artifacts.
                         Clean and annotated versions of the same frame are often published as a pair and may differ only by a few drawn lines. Look closely before choosing image.
  spectrum_or_plot     : data on axes — spectra, light curves, transmission curves.
  comparison_composite : TWO OR MORE SEPARATE PANELS presented together in one graphic — Webb beside Hubble, NIRCam beside MIRI, before/after, a strip of framed insets. The test is whether you can see distinct panel boundaries. A single picture that BLENDS data from several telescopes into one frame is NOT this — "Chandra Adds X-ray Vision to Webb Images" is one merged picture, so it is image. NASA says "composite" to mean blended; here it means panelled.
If both annotated and a comparison, choose comparison_composite.
MODALITY IS DECIDED BY THE IMAGE, NOT THE CAPTION. Count the panels you can actually see in the picture. IGNORE the words "composite", "combined", "merged", "blended" and "multi-observatory" wherever they appear in the text — they describe how the DATA was processed, not how the GRAPHIC is laid out. Webb and Chandra data merged into a single frame is a single frame: that is image, not comparison_composite. Only visible panel divisions — separate sub-pictures with their own borders — make it comparison_composite.

STEP 4 — OBJECT NAME. The catalogue designation or proper name of the primary object, copied VERBATIM from the title or caption — "NGC 6334", "WASP-96 b", "SMACS 0723", "Cassiopeia A". Copy exactly; do not normalise, expand, or correct. Prefer the catalogue designation when both appear. Use null when no specific object is named. Never invent one.

STEP 5 — INSTRUMENT. NIRCam, MIRI, NIRSpec, NIRISS, FGS, multiple, non-Webb, or unknown.
  - unknown IS THE CORRECT ANSWER whenever the text does not name an instrument. It is not a failure or a hedge, and most captions simply never say. Never infer an instrument from the subject, from how the image looks, or from the kind of observation.
  - multiple requires TWO OR MORE Webb instruments named explicitly. One named instrument means that instrument, not multiple.
  - The instrument is usually in the title in parentheses: "(NIRCam Image)", "(NIRSpec MSA Emission Spectra)", "(NIRCam and MIRI)".
  - non-Webb when the data is entirely from another observatory. For a Webb + Chandra composite, name the Webb instrument if stated, else unknown.

STEP 6 — CONFIDENCE and RATIONALE. Use low when the caption is thin or the case is genuinely contested, not merely when the object is unfamiliar. Keep the rationale under 20 words.

Respond only with the JSON object."""

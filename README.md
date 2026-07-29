# Cell Segmentation and Tracking

Automated single-cell tracking for any cell culture time-lapse images.
Currently validated end-to-end on HemaChip HSEC microfluidic culture data (patient CB007)
as a test case - the core engine itself is cell-type agnostic and does not assume
HemaChip-specific biology anywhere.

## Architecture

```
config.py             Single source of truth for all configuration and data
                       records (SpotRecord, TrackRecord, PopulationRecord, etc.)

core/                  Cell-type agnostic engine - works on any cell culture
  input_handler.py       Reads any TIF folder -> FieldOfView objects, handles
                          multiple filename conventions, applies frame_mode
                          selection (all / first / median / clinical)
  segmentor.py            Cellpose wrapper - GPU/CPU/parallel execution,
                           handles both Cellpose v2 and v3 APIs
  spot_features.py          Per-cell per-frame morphology (area, perimeter,
                             circularity, eccentricity, solidity, intensity) -
                             pure NumPy, computed from the segmentation mask
  tracker.py                 Single-cell tracking - laptrack LAP algorithm
                              (same algorithm family as TrackMate) by default,
                              with a nearest-centroid fallback/alternative mode.
                              Also computes each tracked cell's lifetime summary
                              (lifespan, velocity, displacement, confinement
                              ratio, size/shape change)
  track_features.py           Per ROI level metrics (mean/std lifespan, cell counts
                               over time, arrival/departure rate)
  viability.py                  Diameter-based viability classification -
                                 implemented but INACTIVE by default; requires
                                 a validated threshold before use (see below)
  exporter.py                    Atomic CSV writes (spots/tracks/population)
                                  + run_config.json + run_summary.json
  visualizer.py                    QC segmentation overlays (raw image next to
                                    Cellpose outlines) and  track trajectory
                                    overlays.
  analytics.py                      ROI-level analytics dashboard - one chart
                                     per PNG file (line/pie/box plots, spatial
                                     heatmaps, chip-condition comparisons),
                                     generated automatically for any run with
                                     2+ processed FOVs
  spot_loader.py                     Reloads a prior run's spots.csv, enabling
                                      retrack.py's fast re-tracking without
                                      re-running segmentation

plugins/                Experiment-specific logic - never touches core/
  base_plugin.py           Abstract interface (enrich_fovs, compute_fov_features,
                            export) that every plugin implements
  standard/plugin.py         No-op default for generic, non-HemaChip cell culture
  hemachip/                    HemaChip HSEC experiment plugin
    epf_parser.py                 Parses HemaChip .epf protocol XML files -
                                   capture interval, chip layout, boustrophedon
                                   (snake) scan pattern, multi-chip detection
    scanner.py                      Matches ROI folders on disk to EPF grid/
                                     chip metadata, tolerant of partial
                                     downloads and missing EPF files
    clinical_features.py               The 4 HemaChip clinical features:
                                        adhesion_rate, spreading_rate,
                                        endpoint_value, endpoint_variability
    plugin.py                            Wires the above into the pipeline via
                                          the BasePlugin hooks

models/                 Model registry - where a fine-tuned Cellpose model
                         (e.g. hsec_v1) is placed once fine-tuning is complete.
                         Currently empty; pipeline uses stock cyto3 by default.

pipeline.py             Main CLI entry point - full run from raw images to
                         CSVs + visualisations + analytics
retrack.py              Re-runs tracking (and everything downstream) from a
                         prior run's spots.csv, without re-segmenting - useful
                         for sweeping tracking parameters cheaply (seconds
                         instead of the 30-60+ minutes a full segmentation
                         pass takes)
job_manager.py           Checkpointing + resume support for long unattended
                          runs (e.g. Colab). Checkpoint identity is derived
                          deterministically from --output_dir, so re-running
                          the same command against the same output resumes
                          correctly.

tests/                  51 tests across 6 files, synthetic data only (no
                         Cellpose/GPU required to run the suite)
  test_core.py              Core engine: input handling, frame selection,
                             segmentation feature extraction, tracking,
                             population stats, export, config, resume
  test_hemachip.py            EPF parsing (against a real CB007 fixture file),
                               scanner, clinical features, full plugin
                               integration
  test_retrack.py               spot_loader round-trip correctness and speed
  test_analytics.py               Chart generation across generic/gradient/
                                   multi-condition scenarios
  test_viability.py                 Classification correctness, inactive-by-
                                     default behaviour, summary statistics
  test_visualizer.py                  QC/track overlay color generation
```

## Install

```bash
pip install -r requirements.txt
```

## Run the test suite

Fast (seconds), synthetic data only - no Cellpose model or GPU needed:

```bash
python tests/test_core.py
python tests/test_hemachip.py
python tests/test_retrack.py
python tests/test_analytics.py
python tests/test_viability.py
python tests/test_visualizer.py
```

Each should end with `Results: N passed, 0 failed`.

## Quick start - single field of view

```bash
python pipeline.py \
  --input_dir  /path/to/ROI-1 \
  --output_dir /path/to/results \
  --pixel_size_um 0.3769 \
  --time_interval_min 5.0 \
  --model cyto3 \
  --diameter 31.9 \
  --cellprob_threshold 0.3 \
  --frame_mode all \
  --save_masks --save_overlays --save_tracks_viz
```

## Batch mode - multiple fields of view (HemaChip session)

Point `--input_dir` at the session folder (containing the `.epf` file and
`ROI-*` subfolders), not a single ROI folder, and add `--experiment hemachip`
to activate grid/chip metadata and the 4 clinical features:

```bash
python pipeline.py \
  --input_dir  /path/to/session_folder \
  --output_dir /path/to/results \
  --pixel_size_um 0.3769 \
  --time_interval_min 5.0 \
  --model cyto3 \
  --diameter 31.9 \
  --cellprob_threshold 0.3 \
  --frame_mode all \
  --max_dist_px 20 \
  --experiment hemachip \
  --save_masks --save_overlays --save_tracks_viz \
  --gpu
```

`--fov_pattern` can restrict a batch run to specific ROIs, e.g.
`--fov_pattern "ROI-[12]"` for just ROI-1 and ROI-2.

## Resume an interrupted run

```bash
python pipeline.py \
  --input_dir  /path/to/data \
  --output_dir /path/to/results \
  --resume
```

Re-running the exact same command against the same `--output_dir` (with
`--resume`) skips any FOV already checkpointed and continues from where it
left off. If a checkpoint exists and `--resume` is not passed, the pipeline
refuses to proceed rather than silently redoing completed work.

## Sweeping tracking parameters without re-segmenting

Segmentation is the overwhelming majority of total runtime in every real run
so far. Once `spots.csv` exists from any completed run, `retrack.py` re-runs
tracking (and everything downstream: population stats, clinical features,
export) in seconds instead of re-running Cellpose:

```bash
python retrack.py \
  --prior_output_dir /path/to/previous/run \
  --output_dir /path/to/new_results \
  --max_dist_px 20 \
  --experiment hemachip \
  --raw_input_dir /path/to/original/session_folder \
  --save_tracks_viz
```

`--raw_input_dir` is optional; without it, tracking and clinical-feature
values are unaffected, but grid metadata (grid_row/grid_col) and track
trajectory overlays are skipped, since those need the original `.epf` file
or raw images respectively.

`--track_min_elapsed_min` restricts tracking to frames after a given elapsed
time, without affecting segmentation-derived cell counts. On the real CB007
dataset this was tested at 0/15/30/60/120 minutes and made no measurable
difference to tracking quality - `--max_dist_px` alone was found to correctly
exclude implausible frame-to-frame matches regardless of cutoff. The flag is
kept available since a different dataset's motion characteristics could behave
differently; it is not needed for CB007 specifically.

## Output files

| File / folder | Description |
|---|---|
| `spots.csv` | Per cell, per frame (20 fields: position, area, circularity, eccentricity, solidity, intensity, equivalent diameter, viability, track ID) |
| `tracks.csv` | Per cell, whole tracked lifetime (27 fields: lifespan, displacement, velocity, confinement ratio, size/shape change) |
| `population.csv` | Per field of view (27+ fields: cell counts over time, track survival rate, arrival/departure rate; HemaChip plugin adds grid position and the 4 clinical features) |
| `results_chip_clinical.csv` | Per chip (HemaChip plugin only) - mean/std/min/max/median of the 4 clinical features across all ROIs on that chip |
| `run_config.json` | Full configuration for the run, for reproducibility |
| `run_summary.json` | Timing breakdown and error log |
| `masks/` | Cellpose segmentation mask TIFs, one per processed frame |
| `qc_overlays/<ROI>/` | Segmentation QC overlay images (first/middle/last frame) and track trajectory overlay |
| `qc_summary_grid.png` | At-a-glance grid of every processed FOV's cell count, flagging zero-cell and unusually-high-density ROIs (batch runs with 2+ FOVs only) |
| `analytics/` | ROI-level charts - one PNG per metric per chart type (trend lines, distributions, spatial heatmaps, chip comparisons); generated automatically for any run with 2+ processed FOVs |
| `per_fov/<ROI>/` | Per-FOV intermediate spots/tracks CSVs |
| `checkpoints/` | Resume state (JSON + pickled per-FOV results) |

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `--model` | `cyto3` | Cellpose model name (swap in a fine-tuned model, e.g. `hsec_v1`, once available) |
| `--diameter` | auto | Cell diameter in pixels. Auto-falls back to 31.9px when `--pixel_size_um` matches the validated HemaChip setup (0.3769), otherwise estimates from a generic 10um cell |
| `--cellprob_threshold` | `0.0` | Cellpose detection confidence cutoff - raise to reduce false positives (debris) |
| `--flow_threshold` | `0.4` | Cellpose shape-acceptance strictness - raise to reduce merged-cell errors |
| `--min_diam_um` / `--max_diam_um` | `4.0` / `30.0` | Post-segmentation size filter (um), discards implausible detections regardless of what Cellpose reported |
| `--viability_diameter_um` | none (inactive) | Diameter threshold below which a cell is classified non-viable. Deliberately not defaulted to the source publication's own value - see `core/viability.py` for why |
| `--frame_mode` | `all` | `all` / `first` / `median` / `clinical` (18 biologically-timed timepoints) |
| `--tracker_method` | `lap` | `lap` (laptrack, default) or `nearest` (simple radius match, matches the published HSEC methodology exactly, for direct comparability) |
| `--max_dist_px` | `50.0` | Max pixel distance between frames for two detections to be linked as the same cell |
| `--max_dist_um` | none | Same as above, in microns (converted using `--pixel_size_um`) - use this to match a physical distance such as the published 150um |
| `--gap_dist_px` / `--gap_frames` | `80.0` / `2` | Distance and frame-count allowance for reconnecting a cell that briefly disappeared |
| `--min_track_frames` | `3` | Minimum track length to be reported; shorter tracks are treated as untracked noise |
| `--track_min_elapsed_min` | none | Excludes frames before this elapsed time from tracking only (segmentation/cell counts unaffected) |
| `--experiment` | `standard` | `standard` (no-op) or `hemachip` (activates grid metadata + clinical features) |
| `--resume` | off | Continue from a checkpoint at the same `--output_dir` |
| `--fov_pattern` | `ROI-*` | Glob pattern restricting which FOV subfolders are processed in batch mode |
| `--gpu` | off | Use GPU for Cellpose (forces single-process mode) |

Run `python pipeline.py --help` or `python retrack.py --help` for the
complete, current list - both are argparse-based and this table is not
guaranteed to stay exhaustive as flags are added.

## Design principles

- Cell-type agnostic core - works on any cell culture; HSEC is the current test case, not the target
- Experiment-specific logic lives entirely in `plugins/`, never in `core/` 
- Checkpointing after every FOV, with deterministic resume identity - safe for long unattended Colab runs
- laptrack (pure Python LAP tracker) instead of TrackMate/Fiji - no Java dependency, cloud deployable; same underlying algorithm

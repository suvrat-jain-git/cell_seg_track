from __future__ import annotations
from pathlib import Path
from enum import Enum
import json

class FrameMode(str, Enum):
    ALL = "all"
    FIRST = "first"
    MEDIAN = "median"
    CLINICAL = "clinical"

class ExperimentType(str, Enum):
    STANDARD = "standard"
    HEMACHIP = "hemachip"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class SegmentationConfig:
    def __init__(self, model_name="cyto3", diameter=None, use_gpu=False,
                 flow_threshold=0.4, cellprob_threshold=0.0,
                 min_diameter_um=4.0, max_diameter_um=30.0, batch_size=8):
        assert min_diameter_um < max_diameter_um
        self.model_name = model_name
        self.diameter = diameter
        self.use_gpu = use_gpu
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_diameter_um = min_diameter_um
        self.max_diameter_um = max_diameter_um 
        self.batch_size = batch_size

    def to_dict(self):
        return dict(model_name=self.model_name, diameter=self.diameter, use_gpu=self.use_gpu,
                    flow_threshold=self.flow_threshold, cellprob_threshold=self.cellprob_threshold,
                    min_diameter_um=self.min_diameter_um, max_diameter_um=self.max_diameter_um,
                    batch_size=self.batch_size)

class TrackingConfig:
    def __init__(self, max_distance_px=50.0, gap_closing_max_dist_px=80.0,
                 gap_closing_max_frames=2, min_track_length_frames=3,
                 detect_division=False, division_max_dist_px=40.0,
                 tracker_method="lap", min_elapsed_min=None):
        assert max_distance_px > 0
        assert min_track_length_frames >= 2
        assert tracker_method in ("lap", "nearest")
        self.max_distance_px = max_distance_px
        self.gap_closing_max_dist_px = gap_closing_max_dist_px
        self.gap_closing_max_frames = gap_closing_max_frames
        self.min_track_length_frames = min_track_length_frames
        self.detect_division = detect_division
        self.division_max_dist_px = division_max_dist_px
        self.tracker_method = tracker_method
        self.min_elapsed_min = min_elapsed_min

    def to_dict(self):
        return dict(max_distance_px=self.max_distance_px,
                    gap_closing_max_dist_px=self.gap_closing_max_dist_px,
                    gap_closing_max_frames=self.gap_closing_max_frames,
                    min_track_length_frames=self.min_track_length_frames,
                    detect_division=self.detect_division,
                    division_max_dist_px=self.division_max_dist_px,
                    tracker_method=self.tracker_method,
                    min_elapsed_min=self.min_elapsed_min)

class ImagingConfig:
    def __init__(self, pixel_size_um=None, time_interval_min=None,
                 expected_width_px=1600, expected_height_px=1600, is_timelapse=True):
        self.pixel_size_um = pixel_size_um
        self.time_interval_min = time_interval_min
        self.expected_width_px = expected_width_px
        self.expected_height_px = expected_height_px
        self.is_timelapse = is_timelapse

    def to_dict(self):
        return dict(pixel_size_um=self.pixel_size_um, time_interval_min=self.time_interval_min,
                    expected_width_px=self.expected_width_px, expected_height_px=self.expected_height_px,
                    is_timelapse=self.is_timelapse)

class OutputConfig:
    def __init__(self, save_masks=True, save_overlays=True, save_tracks_viz=True,
                 overlay_max_fovs=25, compress_masks=False):
        self.save_masks = save_masks
        self.save_overlays = save_overlays
        self.save_tracks_viz = save_tracks_viz
        self.overlay_max_fovs = overlay_max_fovs
        self.compress_masks = compress_masks

    def to_dict(self):
        return dict(save_masks=self.save_masks, save_overlays=self.save_overlays,
                    save_tracks_viz=self.save_tracks_viz, overlay_max_fovs=self.overlay_max_fovs,
                    compress_masks=self.compress_masks)

class PipelineConfig:
    def __init__(self, input_dir, output_dir,
                 segmentation=None, tracking=None, imaging=None, output=None,
                 frame_mode=None, experiment_type=None,
                 n_workers=1, log_level=None, checkpoint_dir=None,
                 run_id="", pipeline_version="1.0.0"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        self.segmentation = segmentation or SegmentationConfig()
        self.tracking = tracking or TrackingConfig()
        self.imaging = imaging or ImagingConfig()
        self.output = output or OutputConfig()
        self.frame_mode = frame_mode or FrameMode.ALL
        self.experiment_type = experiment_type or ExperimentType.STANDARD
        self.n_workers = n_workers
        self.log_level = log_level or LogLevel.INFO
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.output_dir / "checkpoints"
        self.run_id = run_id
        self.pipeline_version = pipeline_version

    def to_json(self, path):
        d = dict(input_dir=str(self.input_dir), output_dir=str(self.output_dir),
                 frame_mode=self.frame_mode.value, experiment_type=self.experiment_type.value,
                 n_workers=self.n_workers, log_level=self.log_level.value,
                 run_id=self.run_id, pipeline_version=self.pipeline_version,
                 segmentation=self.segmentation.to_dict(), tracking=self.tracking.to_dict(),
                 imaging=self.imaging.to_dict(), output=self.output.to_dict())
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(input_dir=d["input_dir"], output_dir=d["output_dir"],
                   frame_mode=FrameMode(d["frame_mode"]), experiment_type=ExperimentType(d["experiment_type"]),
                   n_workers=d["n_workers"], log_level=LogLevel(d["log_level"]),
                   segmentation=SegmentationConfig(**d["segmentation"]),
                   tracking=TrackingConfig(**d["tracking"]),
                   imaging=ImagingConfig(**d["imaging"]), output=OutputConfig(**d["output"]))

class FrameInfo:
    def __init__(self, path, frame_index, elapsed_min, fov_id=""):
        self.path = Path(path)
        self.frame_index = frame_index
        self.elapsed_min = elapsed_min
        self.fov_id = fov_id

class FieldOfView:
    def __init__(self, path, fov_id, frames, pixel_size_um=0.377,
                 time_interval_min=5.0, grid_row=0, grid_col=0,
                 x_mm=0.0, y_mm=0.0, condition=""):
        self.path = Path(path)
        self.fov_id = fov_id
        self.frames = frames
        self.pixel_size_um = pixel_size_um
        self.time_interval_min = time_interval_min
        self.grid_row = grid_row
        self.grid_col = grid_col
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.condition = condition

    @property
    def n_frames(self):
        return len(self.frames)

    @property
    def is_empty(self):
        return len(self.frames) == 0

    @property
    def duration_min(self):
        if not self.frames:
            return 0.0
        return self.frames[-1].frame_index * self.time_interval_min

class SpotRecord:
    def __init__(self, fov_id, frame_index, elapsed_min, spot_id, track_id=-1,
                 centroid_x_px=0.0, centroid_y_px=0.0, centroid_x_um=0.0, centroid_y_um=0.0,
                 area_px=0.0, area_um2=0.0, perimeter_px=0.0, circularity=0.0,
                 eccentricity=0.0, mean_intensity=0.0, solidity=0.0,
                 equivalent_diameter_um=0.0, is_viable=None,
                 segmentation_ok=True, error_msg=""):
        self.fov_id = fov_id
        self.frame_index = frame_index
        self.elapsed_min = elapsed_min
        self.spot_id = spot_id
        self.track_id = track_id
        self.centroid_x_px = centroid_x_px
        self.centroid_y_px = centroid_y_px
        self.centroid_x_um = centroid_x_um
        self.centroid_y_um = centroid_y_um
        self.area_px = area_px
        self.area_um2 = area_um2
        self.perimeter_px = perimeter_px
        self.circularity = circularity
        self.eccentricity = eccentricity
        self.mean_intensity = mean_intensity
        self.solidity = solidity
        self.equivalent_diameter_um = equivalent_diameter_um
        self.is_viable = is_viable
        self.segmentation_ok = segmentation_ok
        self.error_msg = error_msg

    def to_dict(self):
        return dict(fov_id=self.fov_id, frame_index=self.frame_index, elapsed_min=self.elapsed_min,
                    spot_id=self.spot_id, track_id=self.track_id, centroid_x_px=self.centroid_x_px,
                    centroid_y_px=self.centroid_y_px, centroid_x_um=self.centroid_x_um,
                    centroid_y_um=self.centroid_y_um, area_px=self.area_px, area_um2=self.area_um2,
                    perimeter_px=self.perimeter_px, circularity=self.circularity,
                    eccentricity=self.eccentricity, mean_intensity=self.mean_intensity,
                    solidity=self.solidity, equivalent_diameter_um=self.equivalent_diameter_um,
                    is_viable=self.is_viable, segmentation_ok=self.segmentation_ok, error_msg=self.error_msg)

class TrackRecord:
    def __init__(self, fov_id, track_id, first_frame=0, last_frame=0,
                 first_elapsed_min=0.0, last_elapsed_min=0.0,
                 lifespan_frames=0, lifespan_min=0.0, n_gaps=0,
                 arrived_in_frame=True, survived_to_end=False,
                 total_displacement_px=0.0, total_displacement_um=0.0,
                 net_displacement_px=0.0, net_displacement_um=0.0,
                 confinement_ratio=0.0, mean_velocity_um_min=0.0, max_velocity_um_min=0.0,
                 area_first_um2=0.0, area_last_um2=0.0, area_change_um2=0.0,
                 area_change_rate_um2_min=0.0, circularity_first=0.0, circularity_last=0.0,
                 circularity_change=0.0, mean_circularity=0.0, mean_area_um2=0.0):
        self.fov_id = fov_id
        self.track_id = track_id
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.first_elapsed_min = first_elapsed_min
        self.last_elapsed_min = last_elapsed_min
        self.lifespan_frames = lifespan_frames
        self.lifespan_min = lifespan_min
        self.n_gaps = n_gaps
        self.arrived_in_frame = arrived_in_frame
        self.survived_to_end = survived_to_end
        self.total_displacement_px = total_displacement_px
        self.total_displacement_um = total_displacement_um
        self.net_displacement_px = net_displacement_px
        self.net_displacement_um = net_displacement_um
        self.confinement_ratio = confinement_ratio
        self.mean_velocity_um_min = mean_velocity_um_min
        self.max_velocity_um_min = max_velocity_um_min
        self.area_first_um2 = area_first_um2
        self.area_last_um2 = area_last_um2
        self.area_change_um2 = area_change_um2
        self.area_change_rate_um2_min = area_change_rate_um2_min
        self.circularity_first = circularity_first
        self.circularity_last = circularity_last
        self.circularity_change = circularity_change
        self.mean_circularity = mean_circularity
        self.mean_area_um2 = mean_area_um2

    def to_dict(self):
        return dict(self.__dict__)

class PopulationRecord:
    def __init__(self, fov_id, n_frames_total=0, n_frames_processed=0,
                 n_tracks_total=0, n_tracks_survived=0, pct_tracks_survived=0.0,
                 mean_lifespan_min=0.0, std_lifespan_min=0.0, median_lifespan_min=0.0,
                 mean_velocity_um_min=0.0, mean_displacement_um=0.0, mean_confinement_ratio=0.0,
                 mean_area_um2=0.0, mean_circularity=0.0, mean_area_change_um2=0.0,
                 cell_count_first=0, cell_count_last=0, cell_count_max=0, cell_count_mean=0.0,
                 arrival_rate_per_min=0.0, departure_rate_per_min=0.0, net_accumulation_rate=0.0,
                 grid_row=0, grid_col=0, x_mm=0.0, y_mm=0.0, condition=""):
        self.fov_id = fov_id
        self.n_frames_total = n_frames_total
        self.n_frames_processed = n_frames_processed
        self.n_tracks_total = n_tracks_total
        self.n_tracks_survived = n_tracks_survived
        self.pct_tracks_survived = pct_tracks_survived
        self.mean_lifespan_min = mean_lifespan_min
        self.std_lifespan_min = std_lifespan_min
        self.median_lifespan_min = median_lifespan_min
        self.mean_velocity_um_min = mean_velocity_um_min
        self.mean_displacement_um = mean_displacement_um
        self.mean_confinement_ratio = mean_confinement_ratio
        self.mean_area_um2 = mean_area_um2
        self.mean_circularity = mean_circularity
        self.mean_area_change_um2 = mean_area_change_um2
        self.cell_count_first = cell_count_first
        self.cell_count_last = cell_count_last
        self.cell_count_max = cell_count_max
        self.cell_count_mean = cell_count_mean
        self.arrival_rate_per_min = arrival_rate_per_min
        self.departure_rate_per_min = departure_rate_per_min
        self.net_accumulation_rate = net_accumulation_rate
        self.grid_row = grid_row
        self.grid_col = grid_col
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.condition = condition

    def to_dict(self):
        return dict(self.__dict__)

"""Physical constants and default settings.

All lengths are in cm, angles in radians, and times in seconds.
The map dimensions use the real measurements given:
  - Floor dot spacing: 9 cm
  - Walls: 9 cm long x 0.5 cm thick x 2.5 cm tall (placed between two adjacent dots)
The Hamster robot values are approximations of the real hardware; refine them with calibration (`python -m hamster_super.robot.calibrate`).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

# ---- Map (measured values) ----
GRID_SPACING = 9.0       # cm, spacing between floor dots
WALL_LENGTH = 9.0        # cm
WALL_THICKNESS = 0.5     # cm
WALL_HEIGHT = 2.5        # cm


@dataclass
class RobotSpec:
    """Robomation Hamster (approximate)."""
    body_length: float = 4.0         # cm (front-back)
    body_width: float = 3.5          # cm
    body_height: float = 3.0         # cm (approximate; measure the real robot)
    collision_radius: float = 2.3    # cm, circular collision approximation
    track_width: float = 3.0         # cm, distance between the left and right wheels
    max_speed: float = 10.0          # cm/s at wheel command 100 (measured on the real robot)
    motor_tau: float = 0.06          # s, motor response time constant
    ir_range: float = 6.0            # cm, effective range of the IR proximity sensors
    ir_angle: float = math.radians(25)  # IR sensor mounting angle (left/right)
    ir_offset: float = 1.8           # cm, forward offset of the IR sensors from the center


@dataclass
class CameraSpec:
    """Camera on the robot.

    Physical setup: a camera module the same size as the Hamster sits on top
    of it on a 1.5 cm attachment, facing the direction of travel. Its tilt can
    be set freely but stays fixed once set, so `pitch` is a calibration value.
    Lens height = body height (3.0) + attachment (1.5) + half the module (1.5).
    """
    width: int = 160                 # processing resolution
    height: int = 120
    hfov: float = math.radians(70)   # horizontal field of view
    attach_height: float = 1.5       # cm, attachment between the Hamster and the camera module
    module_height: float = 3.0       # cm, camera module (same size as the Hamster)
    mount_height: float = 6.0        # cm, lens height above the floor
    mount_forward: float = 2.0       # cm, lens forward of the robot center (front face of the module)
    pitch: float = math.radians(35)  # downward tilt, set once by hand and then fixed
    n_rays: int = 21                 # number of depth rays the policy sees
    ray_fov: float = math.radians(64)  # angular range covered by the rays (<= hfov)
    max_range: float = 45.0          # cm

    @property
    def focal(self) -> float:
        return (self.width / 2) / math.tan(self.hfov / 2)

    def ray_angles(self):
        import numpy as np
        return np.linspace(self.ray_fov / 2, -self.ray_fov / 2, self.n_rays)  # left(+) -> right(-)


@dataclass
class DomainRandomization:
    """Range of real-world disturbances injected during training."""
    enabled: bool = True
    wheel_gain: tuple = (0.92, 1.08)       # per-wheel speed gain (left/right imbalance)
    max_speed: tuple = (8.5, 11.5)         # cm/s (real top speed 10 cm/s, battery/floor variation)
    motor_tau: tuple = (0.02, 0.15)
    action_delay_steps: tuple = (0, 2)     # camera/communication latency (in steps)
    track_width: tuple = (2.7, 3.3)
    slip_std: float = 0.04                 # per-step speed noise ratio
    heading_drift_std: float = 0.01        # rad/step
    ray_noise_std: float = 0.04            # multiplicative noise on rays
    ray_dropout: float = 0.03
    ray_spurious: float = 0.01             # probability of a false short reading
    ir_noise_std: float = 0.05
    pose_err_pos: float = 2.0              # cm, max localization error shown to the planner
    pose_err_yaw: float = math.radians(4)
    calib_error: float = 0.07              # residual error of the speed/track calibration
    cam_pitch_err: float = math.radians(1.0)  # tilt error left after `calibrate pitch`
    cam_height_err: float = 0.3            # cm, lens height vs. the configured value


@dataclass
class EnvConfig:
    dt: float = 0.1                        # control period (10 Hz)
    substeps: int = 4
    steps_per_cell: int = 40               # timeout = steps_per_cell * shortest path cells + 100
    known_map: bool = False                # True: the planner knows the full map in advance
    full_stack_prob: float = 0.0           # fraction of episodes using odometry + particle filter
    # Reward
    r_progress: float = 0.1                # per cm of progress
    r_goal: float = 10.0
    r_collision: float = -0.5
    r_time: float = -0.02
    r_smooth: float = -0.02
    r_out: float = -5.0
    r_reverse: float = -0.05               # per step of backward driving (the camera cannot see behind)
    robot: RobotSpec = field(default_factory=RobotSpec)
    camera: CameraSpec = field(default_factory=CameraSpec)
    dr: DomainRandomization = field(default_factory=DomainRandomization)


def to_dict(cfg) -> dict:
    return asdict(cfg)


def from_dict(cls, d: dict):
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        default = getattr(cls(), f.name)
        if is_dataclass(default) and isinstance(v, dict):
            v = from_dict(type(default), v)
        elif isinstance(default, tuple) and isinstance(v, list):
            v = tuple(v)
        kwargs[f.name] = v
    return cls(**kwargs)


def save_json(cfg, path):
    Path(path).write_text(json.dumps(to_dict(cfg), indent=2, ensure_ascii=False))


def load_env_config(path) -> EnvConfig:
    return from_dict(EnvConfig, json.loads(Path(path).read_text()))

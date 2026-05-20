"""Minimal configuration package for standalone LensKit model definitions."""

from dataclasses import dataclass, field


@dataclass
class ParallelSettings:
    num_threads: int = 1
    num_backend_threads: int = 1


@dataclass
class MachineSettings:
    pass


@dataclass
class RandomSettings:
    seed: int | None = None


@dataclass
class LenskitSettings:
    parallel: ParallelSettings = field(default_factory=ParallelSettings)
    machine: MachineSettings = field(default_factory=MachineSettings)
    random: RandomSettings = field(default_factory=RandomSettings)


def lenskit_config():
    return LenskitSettings()


def configure(*args, **kwargs):
    return lenskit_config()


def load_config_data(*args, **kwargs):
    raise RuntimeError("LensKit configuration loading is not included in standalone model definitions.")


def locate_configuration_root(*args, **kwargs):
    return None

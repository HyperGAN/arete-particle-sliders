"""Load the shared steering equations without replacing the strict RAW owner."""

from hashlib import sha256
from pathlib import Path
from sys import modules
from threading import Lock
from types import ModuleType
from typing import NamedTuple
from zipfile import ZipFile

CORE_REVISION = "b1f780decc516808cfd3019d5e5b674457161f86"
DOSE_CORE_REVISION = "c4b3655f43ee9b6e9780bb65766dee0d603cb749"
_SOURCE = {
    "adaptive_steering": (10329, "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"),
    "prompt_steering": (3919, "67ee03066e255fe105735820b0edba53757d943b93239b1279de707d23dbae59"),
}
_DOSE_SOURCE = (6026, "4ffd07fa3793bb0d96e9e207dd45e5516a422497def938a7db7fd083ecf600f4")
_LOAD_LOCK = Lock()
_loaded_core = None


class SteeringCore(NamedTuple):
    """The byte-verified steering modules from the separate shared core wheel."""

    adaptive: ModuleType
    prompt: ModuleType


def _read_source(archive, name, size, digest):
    path = f"particle_sliders/{name}.py"
    entry = archive.getinfo(path)
    if entry.file_size != size:
        raise ValueError(f"Shared steering source size differs: {path}")
    source = archive.read(entry)
    if sha256(source).hexdigest() != digest:
        raise ValueError(f"Shared steering source checksum differs: {path}")
    return source


def _load_source(namespace, wheel, name, source):
    # Dataclasses use sys.modules to resolve their Python module. The private
    # names keep the archived particle_sliders package and sys.path in place.
    if namespace not in modules:
        module = ModuleType(namespace)
        module.__file__ = f"{wheel}/particle_sliders/{name}.py"
        modules[namespace] = module
        try:
            exec(compile(source, module.__file__, "exec"), module.__dict__)
        except BaseException:
            del modules[namespace]
            raise
    return modules[namespace]


def load_steering_core(wheel):
    """Read exact standalone #269 modules from a local wheel, without installing it."""
    global _loaded_core
    wheel = Path(wheel).resolve()
    sources = {}
    with ZipFile(wheel) as archive:
        for name, (size, digest) in _SOURCE.items():
            sources[name] = _read_source(archive, name, size, digest)
    with _LOAD_LOCK:
        if _loaded_core is not None:
            return _loaded_core
        _loaded_core = SteeringCore(*(_load_source(f"{__package__}._core_269_{name}", wheel, name, source)
                                      for name, source in sources.items()))
        return _loaded_core


def load_dose_matcher(wheel):
    """Load the exact shared binary32 dose matcher without installing its wheel."""
    wheel = Path(wheel).resolve()
    with ZipFile(wheel) as archive:
        source = _read_source(archive, "strength_schedule", *_DOSE_SOURCE)
    with _LOAD_LOCK:
        return _load_source(f"{__package__}._core_dose", wheel, "strength_schedule", source)


def require_dose_matcher(matcher):
    if matcher is None or matcher is not modules.get(f"{__package__}._core_dose"):
        raise ValueError("Load the shared dose equations with load_dose_matcher")


def require_steering_core(core):
    if _loaded_core is None or core is not _loaded_core:
        raise ValueError("Load the steering equations with load_steering_core")

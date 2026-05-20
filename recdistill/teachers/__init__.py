__all__ = [
    "TeacherSource",
    "TeacherState",
    "PrecomputedScoresScorer",
    "PrecomputedTopKScorer",
    "load_teacher",
    "load_teacher_state",
    "save_teacher_state",
    "inject_static_noise",
]


def __getattr__(name: str):
    if name == "TeacherState":
        from recdistill.teachers.state import TeacherState

        return TeacherState
    if name == "PrecomputedScoresScorer":
        from recdistill.teachers.state import PrecomputedScoresScorer

        return PrecomputedScoresScorer
    if name == "PrecomputedTopKScorer":
        from recdistill.teachers.state import PrecomputedTopKScorer

        return PrecomputedTopKScorer
    if name == "TeacherSource":
        from recdistill.teachers.source import TeacherSource

        return TeacherSource
    if name == "load_teacher":
        from recdistill.teachers.loaders import load_teacher

        return load_teacher
    if name == "load_teacher_state":
        from recdistill.teachers.loaders import load_teacher_state

        return load_teacher_state
    if name == "save_teacher_state":
        from recdistill.teachers.serialization import save_teacher_state

        return save_teacher_state
    if name == "inject_static_noise":
        from recdistill.teachers.noise import inject_static_noise

        return inject_static_noise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

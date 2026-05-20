"""Reusable recommendation distillation framework primitives."""

__all__ = [
    "RecDistillExperimentRunner",
    "STUDENT_CHECKPOINT_FORMAT",
    "TEACHER_FORMAT_VERSION",
    "PrecomputedScoresScorer",
    "PrecomputedTopKScorer",
    "TeacherSource",
    "TeacherState",
    "canonical_distiller_name",
    "canonical_model_name",
    "load_student_checkpoint",
    "load_teacher",
    "load_teacher_state",
    "save_student_checkpoint",
    "save_teacher_state",
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
    if name == "RecDistillExperimentRunner":
        from recdistill.experiment_runner import RecDistillExperimentRunner

        return RecDistillExperimentRunner
    if name == "STUDENT_CHECKPOINT_FORMAT":
        from recdistill.checkpointing import STUDENT_CHECKPOINT_FORMAT

        return STUDENT_CHECKPOINT_FORMAT
    if name == "canonical_distiller_name":
        from recdistill.registry import canonical_distiller_name

        return canonical_distiller_name
    if name == "canonical_model_name":
        from recdistill.registry import canonical_model_name

        return canonical_model_name
    if name == "load_teacher_state":
        from recdistill.teachers.loaders import load_teacher_state

        return load_teacher_state
    if name == "load_teacher":
        from recdistill.teachers.loaders import load_teacher

        return load_teacher
    if name == "save_student_checkpoint":
        from recdistill.checkpointing import save_student_checkpoint

        return save_student_checkpoint
    if name == "load_student_checkpoint":
        from recdistill.checkpointing import load_student_checkpoint

        return load_student_checkpoint
    if name == "save_teacher_state":
        from recdistill.teachers.serialization import save_teacher_state

        return save_teacher_state
    if name == "TEACHER_FORMAT_VERSION":
        from recdistill.teachers.serialization import TEACHER_FORMAT_VERSION

        return TEACHER_FORMAT_VERSION
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

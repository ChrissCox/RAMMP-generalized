"""Learned skills: model-written Python run in a jail whose only effect is requesting guarded primitive calls.

See docs/design/15-learned-skills.md. gate checks the source, jail runs it (bubblewrap, restricted builtins,
a pipe to the host), api validates every primitive call the host receives, library keeps skills and their record.
"""
from .api import PRIMITIVES, PrimitiveError, validate_call
from .gate import GateError, check_source
from .jail import JailError, SkillRun, run_skill
from .library import SkillLibrary

__all__ = ["PRIMITIVES", "GateError", "JailError", "PrimitiveError", "SkillLibrary", "SkillRun", "check_source",
           "run_skill", "validate_call"]

"""CUDA backend boundary reserved for a later implementation."""


def availability() -> tuple[bool, str]:
    return False, "cuda backend is not implemented in Phase 1"

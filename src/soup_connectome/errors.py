class ConnectomeError(Exception):
    """Base class for expected runtime failures."""


class ConfigurationError(ConnectomeError):
    """The requested runtime configuration is invalid."""


class GraphFormatError(ConnectomeError):
    """A graph artifact is malformed or uses an unsupported format."""


class GraphValidationError(GraphFormatError):
    """A graph has internally inconsistent or out-of-range values."""


class ChecksumMismatchError(GraphFormatError):
    """An artifact file differs from its manifest checksum."""


class ArtifactExistsError(ConnectomeError):
    """Writing an artifact would overwrite an existing directory."""


class PathContainmentError(ConnectomeError):
    """A path escapes the artifact directory."""


class FixedPointOverflowError(ConnectomeError):
    """A canonical signed int32 operation would overflow."""


class BackendError(ConnectomeError):
    """Base class for backend resolution and execution errors."""


class BackendNotImplementedError(BackendError):
    """The requested backend is a known but unimplemented boundary."""


class BackendUnavailableError(BackendError):
    """The requested backend exists but is unavailable on this host."""

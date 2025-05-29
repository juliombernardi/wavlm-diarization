# Standard library imports
from dataclasses import dataclass, field
from typing import Annotated


@dataclass
class WavLMModels:
    """WavLMModels is a data class that stores configuration parameters
    for WavLM model usage.

    This class provides a list of allowed model names and a selected
    model name for audio processing tasks.

    Parameters
    ----------
    allowed : List[str], optional
        A list of allowed WavLM model names.
    selected : str, optional
        The currently selected WavLM model name.

    Attributes
    ----------
    allowed : List[str]
        A list of allowed WavLM model names.
    selected : str
        The currently selected WavLM model name.

    Examples
    --------
    >>> models = WavLMModels(
    ...     allowed=["microsoft/wavlm-base", "microsoft/wavlm-base-plus-sv"]
    ... )
    >>> models.allowed
    ['microsoft/wavlm-base', 'microsoft/wavlm-base-plus-sv']
    >>> models.selected
    'microsoft/wavlm-base-plus-sv'

    """

    allowed: Annotated[list[str], 'List of allowed WavLM model names'] = field(default_factory=list)
    selected: Annotated[str, 'Currently selected WavLM model name'] = 'microsoft/wavlm-base-plus-sv'

    def __post_init__(self) -> None:
        """Validate the model names after initialization."""
        if not isinstance(self.allowed, list):
            raise TypeError("Expected 'allowed' to be a list of strings.")
        if any(not isinstance(m, str) for m in self.allowed):
            raise TypeError("All items in 'allowed' must be strings.")

        if not isinstance(self.selected, str):
            raise TypeError("Expected 'selected' to be a string.")


@dataclass
class ModelConfig:
    """Data class that wraps the WavLMModels configuration."""

    wavlm: Annotated[
        WavLMModels,
        'Configuration object for WavLM models',
    ] = field(default_factory=WavLMModels)

    def __post_init__(self) -> None:
        """Validate that the 'wavlm' field is an instance of WavLMModels."""
        if not isinstance(self.wavlm, WavLMModels):
            raise TypeError("Expected 'wavlm' to be an instance of WavLMModels.")


@dataclass
class RuntimeConfig:
    """Runtime configuration (device list, CUDA allocation, etc.)."""

    device: Annotated[list[str], 'List of devices in priority order'] = field(
        default_factory=lambda: ['cuda', 'cpu']
    )
    cuda_alloc_conf: Annotated[str, 'CUDA memory allocation configuration'] = (
        'expandable_segments:True'
    )

    def __post_init__(self) -> None:
        """Validate runtime fields after initialization."""
        if not isinstance(self.device, list):
            raise TypeError("Expected 'device' to be a list of strings.")
        if any(not isinstance(dev, str) for dev in self.device):
            raise TypeError("All items in 'device' must be strings.")

        if not isinstance(self.cuda_alloc_conf, str):
            raise TypeError("Expected 'cuda_alloc_conf' to be a string.")


@dataclass
class RootConfig:
    """Top-level container that integrates runtime and model configurations."""

    runtime: Annotated[
        RuntimeConfig,
        'Runtime configuration settings',
    ] = field(default_factory=RuntimeConfig)

    model: Annotated[
        ModelConfig,
        'Model configuration settings',
    ] = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        """Validate the root-level configuration fields after initialization."""
        if not isinstance(self.runtime, RuntimeConfig):
            raise TypeError("Expected 'runtime' to be a RuntimeConfig instance.")
        if not isinstance(self.model, ModelConfig):
            raise TypeError("Expected 'model' to be a ModelConfig instance.")

"""Configuration loader that reads settings from YAML files."""

from pprint import pprint
import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """Loads and manages configuration settings from YAML files."""

    def __init__(self,
                 agents_config_path: str = "config_agents.yaml",
                 pipeline_config_path: str = "config_pipeline.yaml"):
        """
        Initialize the ConfigLoader with two YAML configuration files.

        Args:
            agents_config_path: Path to the agent-specific YAML configuration file
            pipeline_config_path: Path to the pipeline-specific YAML configuration file
        """
        self.agents_config_path = Path(agents_config_path)
        self.pipeline_config_path = Path(pipeline_config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
        self._process_dynamic_values()

    def _load_config(self) -> None:
        """Load and combine configuration from both YAML files."""
        # Load agents configuration
        if not self.agents_config_path.exists():
            raise FileNotFoundError(f"Agents configuration file not found: {self.agents_config_path}")

        with open(self.agents_config_path, 'r') as f:
            agents_config = yaml.safe_load(f)

        if not agents_config:
            raise ValueError(f"Agents configuration file is empty: {self.agents_config_path}")

        # Load pipeline configuration
        if not self.pipeline_config_path.exists():
            raise FileNotFoundError(f"Pipeline configuration file not found: {self.pipeline_config_path}")

        with open(self.pipeline_config_path, 'r') as f:
            pipeline_config = yaml.safe_load(f)

        if not pipeline_config:
            raise ValueError(f"Pipeline configuration file is empty: {self.pipeline_config_path}")

        # Combine configurations (pipeline config values will override if there are conflicts)
        self._config = {**agents_config, **pipeline_config}

    def _process_dynamic_values(self) -> None:
        """Process dynamic configuration values that depend on other values."""
        # Construct SOURCE_FILE from CODE_POOL if not explicitly set
        if 'SOURCE_FILE' not in self._config and 'CODE_POOL' in self._config:
            code_pool = self._config['CODE_POOL']
            # Extract the base name from code pool (e.g., "deltanet" from "./pool/deltanet")
            pool_name = Path(code_pool).name
            self._config['SOURCE_FILE'] = f"{code_pool}/{pool_name}_base.py"

    def __getattr__(self, name: str) -> Any:
        """
        Allow attribute-style access to configuration values.

        Args:
            name: Configuration key name

        Returns:
            Configuration value

        Raises:
            AttributeError: If configuration key doesn't exist
        """
        if name.startswith('_'):
            # Allow access to private attributes
            return object.__getattribute__(self, name)

        if name in self._config:
            return self._config[name]

        raise AttributeError(f"Configuration key '{name}' not found in configuration files")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with an optional default.

        Args:
            key: Configuration key name
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def reload(self) -> None:
        """Reload configuration from the YAML file."""
        self._load_config()
        self._process_dynamic_values()

    def __repr__(self) -> str:
        return f"ConfigLoader(agents='{self.agents_config_path}', model='{self.model_config_path}')"

    def __str__(self) -> str:
        return f"ConfigLoader using {self.agents_config_path} and {self.model_config_path} with {len(self._config)} settings"


# For backward compatibility, create a default Config instance
Config = ConfigLoader(
    agents_config_path=Path(__file__).parent.resolve() / "config_agents.yaml",
    pipeline_config_path=Path(__file__).parent.resolve() / "config_pipeline.yaml"
)


if __name__ == "__main__":
    # Example usage
    print(Config.__dict__['_config'])

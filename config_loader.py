"""Configuration loader that reads settings from YAML files."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """Loads and manages configuration settings from YAML files."""

    def __init__(self, agents_config_path: str, pipeline_config_path: str, aml_config_path: str) -> None:
        """
        Initialize the ConfigLoader with two YAML configuration files.

        Args:
            agents_config_path: Path to the agent-specific YAML configuration file
            pipeline_config_path: Path to the pipeline-specific YAML configuration file
        """
        self.agents_config_path = Path(agents_config_path)
        self.pipeline_config_path = Path(pipeline_config_path)
        self.aml_config_path = Path(aml_config_path)

        # Store the current folder path
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

        # Load AML configuration
        if not self.aml_config_path.exists():
            raise FileNotFoundError(f"AML configuration file not found: {self.aml_config_path}")

        with open(self.aml_config_path, 'r') as f:
            aml_config = yaml.safe_load(f)

        if not aml_config:
            raise ValueError(f"AML configuration file is empty: {self.aml_config_path}")

        agent_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings', 'architecture', pipeline_config["ARCHITECTURE"], 'agents.yaml')
        # Load agents configuration
        if not os.path.exists(agent_yaml):
            raise FileNotFoundError(f"agents.yaml not found: {agent_yaml}")

        with open(agent_yaml, 'r') as f:
            agent_yaml = yaml.safe_load(f)

        # Combine configurations (pipeline config values will override if there are conflicts)
        self._config = {**agents_config, **pipeline_config, **aml_config, **agent_yaml}

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

    def load_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        Load configuration for a specific agent.

        Args:
            agent_name: Name of the agent to load configuration for
        Returns:
            Dictionary of agent configuration
        Raises:
            ValueError: If agent configuration is not found
        """
        agent_key = f"{agent_name.upper()}"
        if agent_key in self._config:
            return self._config[agent_key]
        raise ValueError(f"Agent configuration '{agent_key}' not found")


# For backward compatibility, create a default Config instance
Config = ConfigLoader(
    agents_config_path=Path(__file__).parent.resolve() / "config_agents.yaml",
    pipeline_config_path=Path(__file__).parent.resolve() / "config_pipeline_deltanet.yaml",
    aml_config_path=Path(__file__).parent.resolve() / "config_azuremachinelearning.yaml"
)


if __name__ == "__main__":
    # Example usage
    for key, value in Config.__dict__['_config'].items():
        value_str = str(value)
        print(f"{key}: {value_str[:100].replace(chr(10), ' ')}...")

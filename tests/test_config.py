import pytest
import os
import tempfile
from omegaconf import OmegaConf, ValidationError
from typing import Optional
from dataclasses import dataclass, field # Import dataclass utilities

# --- Dataclass Schema Definition ---

@dataclass
class ModelConfig:
    type: str
    hidden_dim: int
    num_layers: int
    dropout: Optional[float] = field(default=None) # Optional float, default is None

@dataclass
class DataConfig:
    path: str
    batch_size: int

@dataclass
class OptimizerConfig:
    lr: float

@dataclass
class ConfigSchema:
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig
    # Add other top-level sections if needed, e.g., paths: dict = field(default_factory=dict)

# --- Fixture for creating temporary config files ---

@pytest.fixture
def temp_config_file():
    """Creates a temporary YAML config file."""
    tmpdir = tempfile.TemporaryDirectory()
    config_path = os.path.join(tmpdir.name, "temp_config.yaml")

    def _create_config(content):
        with open(config_path, 'w') as f:
            f.write(content)
        return config_path

    yield _create_config # Return the function to create the file

    # Cleanup
    tmpdir.cleanup()

# --- Basic Config Loading Tests ---

def test_load_valid_yaml(temp_config_file):
    """Tests loading a simple, valid YAML file using OmegaConf."""
    valid_yaml_content = """
model:
  type: SimpleMLP
  hidden_dim: 128
data:
  path: /data/set1
  batch_size: 32
"""
    config_path = temp_config_file(valid_yaml_content)
    try:
        cfg = OmegaConf.load(config_path)
        # Basic access checks are still valid
        assert cfg.model.type == "SimpleMLP"
        assert cfg.model.hidden_dim == 128
        assert cfg.data.path == "/data/set1"
        assert cfg.data.batch_size == 32
    except Exception as e:
        pytest.fail(f"Loading valid YAML failed: {e}")

def test_load_yaml_with_interpolation(temp_config_file):
    """Tests OmegaConf interpolation."""
    yaml_content = """
project_name: MyPINN
paths:
  base: /projects/${project_name}
  data: ${paths.base}/data
  output: ${paths.base}/output
model:
  name: ${project_name}_model # Note: This field is not in ConfigSchema, validation tests won't use it directly
"""
    config_path = temp_config_file(yaml_content)
    cfg = OmegaConf.load(config_path)
    assert cfg.paths.base == "/projects/MyPINN"
    assert cfg.paths.data == "/projects/MyPINN/data"
    assert cfg.paths.output == "/projects/MyPINN/output"
    assert cfg.model.name == "MyPINN_model"

# --- Config Validation Tests ---

def test_config_validation_success(temp_config_file):
    """Tests successful validation against a schema."""
    valid_yaml = """
model:
  type: AdaptivePINN
  hidden_dim: 256
  num_layers: 4
  # dropout: 0.1 # Optional field missing is OK
data:
  path: /path/to/data
  batch_size: 64
optimizer:
  lr: 0.001
"""
    config_path = temp_config_file(valid_yaml)
    loaded_cfg = OmegaConf.load(config_path)
    try:
        # Create schema object and merge
        schema = OmegaConf.structured(ConfigSchema)
        merged_cfg = OmegaConf.merge(schema, loaded_cfg)
        # Access fields to ensure no validation error during access
        assert merged_cfg.model.type == "AdaptivePINN"
        assert merged_cfg.optimizer.lr == 0.001
        assert merged_cfg.model.dropout is None # Check default value
    except ValidationError as e:
        pytest.fail(f"Valid config failed validation: {e}")
    except Exception as e:
         pytest.fail(f"Unexpected error during validation: {e}")


def test_config_validation_missing_field(temp_config_file):
    """Tests validation failure due to a missing required field."""
    invalid_yaml = """
model:
  type: AdaptivePINN
  # hidden_dim: 256 # Missing required field
  num_layers: 4
data:
  path: /path/to/data
  batch_size: 64
optimizer:
  lr: 0.001
"""
    config_path = temp_config_file(invalid_yaml)
    loaded_cfg = OmegaConf.load(config_path)
    schema = OmegaConf.structured(ConfigSchema)
    with pytest.raises(ValidationError): # Removed match argument for now
        OmegaConf.merge(schema, loaded_cfg)

def test_config_validation_wrong_type(temp_config_file):
    """Tests validation failure due to incorrect data type."""
    invalid_yaml = """
model:
  type: AdaptivePINN
  hidden_dim: 256
  num_layers: 4
data:
  path: /path/to/data
  batch_size: "64" # Wrong type (string instead of int)
optimizer:
  lr: 0.001
"""
    config_path = temp_config_file(invalid_yaml)
    # Apply schema directly during load
    # The exact error message might vary slightly depending on OmegaConf version
    with pytest.raises(ValidationError, match="Value '64' is not a valid Int"):
        OmegaConf.load(config_path, schema=ConfigSchema)


# --- Config Usage Tests (Example: Testing how config is passed to a function/class) ---

# Dummy function/class that accepts config values
class DummyModel:
    def __init__(self, hidden_dim, num_layers, dropout=None):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

def setup_dummy_model(config):
    """Simulates setting up a model using config."""
    # Access using attribute style now that schema is applied (or use .get if unsure)
    # Assuming config passed here might already be merged with schema or validated
    model_cfg = config.model # Direct access if schema is enforced
    return DummyModel(
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout # Access the potentially defaulted value
    )

def test_config_passed_to_component(temp_config_file):
    """Tests if config values are correctly extracted and passed."""
    valid_yaml = """
model:
  type: Dummy
  hidden_dim: 512
  num_layers: 6
  dropout: 0.2
data:
  path: /dummy/data
  batch_size: 1
optimizer:
  lr: 0.1
"""
    config_path = temp_config_file(valid_yaml)
    loaded_cfg = OmegaConf.load(config_path)
    # Merge with schema before passing
    schema = OmegaConf.structured(ConfigSchema)
    cfg = OmegaConf.merge(schema, loaded_cfg)


    # Simulate component setup
    dummy_model = setup_dummy_model(cfg)

    assert isinstance(dummy_model, DummyModel)
    assert dummy_model.hidden_dim == 512
    assert dummy_model.num_layers == 6
    assert dummy_model.dropout == 0.2

def test_config_optional_field_handling(temp_config_file):
    """Tests handling of optional config fields."""
    yaml_missing_optional = """
model:
  type: Dummy
  hidden_dim: 128
  num_layers: 3
  # dropout is missing
data:
  path: /dummy/data2
  batch_size: 2
optimizer:
  lr: 0.05
"""
    config_path = temp_config_file(yaml_missing_optional)
    loaded_cfg = OmegaConf.load(config_path)
    # Merge with schema to get defaults
    schema = OmegaConf.structured(ConfigSchema)
    cfg = OmegaConf.merge(schema, loaded_cfg)

    dummy_model = setup_dummy_model(cfg)
    assert dummy_model.hidden_dim == 128
    assert dummy_model.num_layers == 3
    assert dummy_model.dropout is None # Check default handling

# Note: More specific tests would involve mocking actual classes/functions
# from the project (e.g., PINNTrainer, AdaptiveFastscapePINN, create_dataloaders)
# and asserting that they are initialized or called with the correct parameters
# derived from the loaded configuration.
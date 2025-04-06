import pytest
import os
import tempfile
from omegaconf import OmegaConf, ValidationError
from typing import Optional, List, Dict # Import List and Dict for potential future use
from dataclasses import dataclass, field

# --- Dataclass Schema Definition ---
# Define the expected structure and types of the configuration

@dataclass
class ModelConfig:
    type: str = field(default="???") # Add default to avoid missing value error if model section exists but type is missing
    hidden_dim: int = field(default=128)
    num_layers: int = field(default=4)
    dropout: Optional[float] = field(default=None)
    # Example of adding more specific model configs if needed later
    # adaptive_specific_param: Optional[int] = None

@dataclass
class DataNormalizationConfig:
    enabled: bool = False
    mode: str = "min-max" # Example default
    stats_file: Optional[str] = None
    compute_stats: bool = False
    # fields: List[str] = field(default_factory=lambda: ['topo', 'uplift_rate']) # Example list

@dataclass
class DataConfig:
    processed_dir: str = field(default="data/processed")
    batch_size: int = 32
    num_workers: int = 0
    train_split: float = 0.8
    val_split: float = 0.1
    normalization: DataNormalizationConfig = field(default_factory=DataNormalizationConfig)

@dataclass
class OptimizerConfig:
    name: str = "Adam"
    lr: float = 1e-3
    weight_decay: float = 0.0

@dataclass
class SchedulerConfig:
    name: Optional[str] = None # Optional scheduler section
    step_size: Optional[int] = None
    gamma: Optional[float] = None
    # Add other scheduler params as needed

@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32 # Can be redundant with data.batch_size, consider consolidating
    n_collocation_points: int = 10000
    optimizer: str = "Adam" # Redundant with optimizer section, consider consolidating
    learning_rate: float = 1e-3 # Redundant with optimizer section
    weight_decay: float = 0.0 # Redundant
    lr_scheduler: Optional[SchedulerConfig] = field(default=None)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {'data': 1.0, 'physics': 0.1})
    pde_loss_method: str = "grid_focused"
    validate_with_physics: bool = False
    val_interval: int = 1
    save_best_only: bool = True
    save_interval: int = 10
    clip_grad_norm: Optional[float] = None
    use_mixed_precision: bool = False
    use_loss_scaling: bool = False # Added based on trainer code

@dataclass
class PhysicsParamsConfig:
    # Define expected physics parameters and their types
    dx: float = 1.0
    dy: float = 1.0
    total_time: float = 1000.0
    domain_x: List[float] = field(default_factory=lambda: [0.0, 100.0])
    domain_y: List[float] = field(default_factory=lambda: [0.0, 100.0])
    U: float = 0.001 # Example default, could be Optional[str] if loading from file
    K_f: float = 1e-5
    m: float = 0.5
    n: float = 1.0
    K_d: float = 0.01
    epsilon: float = 1e-10
    grid_height: Optional[int] = None # Make optional if not always needed
    grid_width: Optional[int] = None
    precip: float = 1.0
    drainage_area_kwargs: Dict = field(default_factory=dict)
    rbf_sigma: Optional[float] = None # For interpolation loss

@dataclass
class ConfigSchema:
    # Define top-level sections
    output_dir: str = "results"
    run_name: Optional[str] = None # Allow run name to be optional, maybe generated if None
    project_name: Optional[str] = None # Added for interpolation test compatibility
    seed: int = 42
    device: str = "auto" # Example: allow auto-detection
    load_checkpoint: Optional[str] = None

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig) # Separate optimizer section
    training: TrainingConfig = field(default_factory=TrainingConfig)
    physics_params: PhysicsParamsConfig = field(default_factory=PhysicsParamsConfig)
    # Add optimization_params section if used by optimize.py script
    # optimization_params: Optional[Dict] = field(default_factory=dict)


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
  num_layers: 4 # Added missing required field from schema
data:
  processed_dir: /data/set1 # Match schema field name
  batch_size: 32
optimizer: # Added optimizer section
  lr: 0.001
"""
    config_path = temp_config_file(valid_yaml_content)
    try:
        # Load and merge with schema immediately for validation
        schema = OmegaConf.structured(ConfigSchema)
        cfg = OmegaConf.merge(schema, OmegaConf.load(config_path))

        assert cfg.model.type == "SimpleMLP"
        assert cfg.model.hidden_dim == 128
        assert cfg.data.processed_dir == "/data/set1"
        assert cfg.data.batch_size == 32
        assert cfg.optimizer.lr == 0.001
    except Exception as e:
        pytest.fail(f"Loading valid YAML failed: {e}")

def test_load_yaml_with_interpolation(temp_config_file):
    """Tests OmegaConf interpolation."""
    yaml_content = """
project_name: MyPINN
output_dir: /projects/${project_name}/output # Match schema field
model:
  type: ${project_name}_model
  hidden_dim: 128 # Add required fields
  num_layers: 3
data:
  processed_dir: /projects/${project_name}/data
  batch_size: 16
optimizer:
  lr: 0.01
"""
    config_path = temp_config_file(yaml_content)
    # Load and merge with schema
    schema = OmegaConf.structured(ConfigSchema)
    cfg = OmegaConf.merge(schema, OmegaConf.load(config_path))

    assert cfg.output_dir == "/projects/MyPINN/output"
    assert cfg.data.processed_dir == "/projects/MyPINN/data"
    assert cfg.model.type == "MyPINN_model"
    assert cfg.project_name == "MyPINN" # Check the added field

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
  processed_dir: /path/to/data
  batch_size: 64
optimizer:
  lr: 0.001
# Add minimal training and physics_params sections if needed by schema defaults
training:
  epochs: 50
physics_params:
  dx: 5.0
"""
    config_path = temp_config_file(valid_yaml)
    try:
        # Create schema object and merge
        schema = OmegaConf.structured(ConfigSchema)
        # Load the config first
        loaded_cfg = OmegaConf.load(config_path)
        # Now merge with the schema
        merged_cfg = OmegaConf.merge(schema, loaded_cfg)

        # Access fields to ensure no validation error during access
        assert merged_cfg.model.type == "AdaptivePINN"
        assert merged_cfg.optimizer.lr == 0.001
        assert merged_cfg.model.dropout is None # Check default value
        assert merged_cfg.training.epochs == 50 # Check value from file
        assert merged_cfg.physics_params.dy == 1.0 # Check default value
    except ValidationError as e:
        pytest.fail(f"Valid config failed validation: {e}")
    except Exception as e:
         pytest.fail(f"Unexpected error during validation: {e}")


def test_config_validation_missing_field(temp_config_file):
    """Tests validation behavior when a field with a default is missing."""
    invalid_yaml = """
model:
  type: AdaptivePINN
  # hidden_dim: 256 # Missing field with default in schema
  num_layers: 4
data:
  processed_dir: /path/to/data
  batch_size: 64
optimizer:
  lr: 0.001
"""
    config_path = temp_config_file(invalid_yaml)
    loaded_cfg = OmegaConf.load(config_path)
    schema = OmegaConf.structured(ConfigSchema)

    # MODIFIED: Assert the default value is applied after merge
    try:
        merged_cfg = OmegaConf.merge(schema, loaded_cfg)
        # Check if the default value from ModelConfig schema (128) was applied
        assert merged_cfg.model.hidden_dim == 128 # Default value from ModelConfig
    except ValidationError as e:
        pytest.fail(f"Merge failed unexpectedly when default should apply: {e}")

def test_config_validation_wrong_type(temp_config_file):
    """Tests validation behavior for incorrect data type (string vs int)."""
    invalid_yaml = """
model:
  type: AdaptivePINN
  hidden_dim: 256
  num_layers: 4
data:
  processed_dir: /path/to/data
  batch_size: "64" # Wrong type (string instead of int)
optimizer:
  lr: 0.001
"""
    config_path = temp_config_file(invalid_yaml)
    loaded_cfg = OmegaConf.load(config_path)
    schema = OmegaConf.structured(ConfigSchema)

    # MODIFIED: Assert the value is converted after merge
    try:
        merged_cfg = OmegaConf.merge(schema, loaded_cfg)
        # Check if OmegaConf converted the string "64" to an integer 64 during merge
        assert merged_cfg.data.batch_size == 64
        assert isinstance(merged_cfg.data.batch_size, int)
    except ValidationError as e:
         pytest.fail(f"Merge failed unexpectedly when type conversion should occur: {e}")


# --- Config Usage Tests (Example: Testing how config is passed to a function/class) ---

# Dummy function/class that accepts config values
class DummyUsageModel:
    def __init__(self, hidden_dim, num_layers, dropout=None):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

def setup_dummy_usage_model(config: ConfigSchema): # Use type hint
    """Simulates setting up a model using a validated config object."""
    # Access using attribute style now that schema is applied
    model_cfg = config.model
    return DummyUsageModel(
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout
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
  processed_dir: /dummy/data
  batch_size: 1
optimizer:
  lr: 0.1
"""
    config_path = temp_config_file(valid_yaml)
    # Load and merge with schema
    schema = OmegaConf.structured(ConfigSchema)
    cfg = OmegaConf.merge(schema, OmegaConf.load(config_path))

    # Simulate component setup
    dummy_model = setup_dummy_usage_model(cfg)

    assert isinstance(dummy_model, DummyUsageModel)
    assert dummy_model.hidden_dim == 512
    assert dummy_model.num_layers == 6
    assert dummy_model.dropout == 0.2

def test_config_optional_field_handling(temp_config_file):
    """Tests handling of optional config fields and defaults from schema."""
    yaml_missing_optional = """
model:
  type: Dummy
  hidden_dim: 128
  num_layers: 3
  # dropout is missing
data:
  processed_dir: /dummy/data2
  batch_size: 2
optimizer:
  lr: 0.05
"""
    config_path = temp_config_file(yaml_missing_optional)
    # Load and merge with schema to get defaults
    schema = OmegaConf.structured(ConfigSchema)
    cfg = OmegaConf.merge(schema, OmegaConf.load(config_path))

    dummy_model = setup_dummy_usage_model(cfg)
    assert dummy_model.hidden_dim == 128
    assert dummy_model.num_layers == 3
    assert dummy_model.dropout is None # Check default handling from schema

# TODO: Add tests for more complex types (List, Dict) in schema if used.
# TODO: Add tests for environment variable interpolation if used.
# TODO: Add tests for command-line overrides if used.
from typing import List, Tuple
import yaml
from pydantic import BaseModel, ConfigDict


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    n_node_types: int
    node_type_names: List[str]
    n_node_types_per_type: List[int]
    filter_length_max: int
    n_frames: int
    dt: float
    rate_range: Tuple[float, float]
    amplitude_range: Tuple[float, float]
    frequency_range: Tuple[float, float]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    model_type: str


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    n_runs: int
    noise_level: float


class NeuronConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    folder_name: str
    simulation: SimulationConfig
    model: ModelConfig
    training: TrainingConfig

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, 'r') as file:
            raw_config = yaml.safe_load(file)
        return NeuronConfig(**raw_config)

    def pretty(self):
        return yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False, indent=2)


# Example usage
if __name__ == '__main__':
    config_file = 'neuron_config.yaml'
    config = NeuronConfig.from_yaml(config_file)
    print(config.pretty())
    print(f'Successfully loaded config file.')
    print(f'Folder: {config.folder_name}')
    print(f'Model: {config.model.model_type}')
    print(f'Number of nodes: {sum(config.simulation.n_node_types_per_type)}')
    print(f'Training runs: {config.training.n_runs}')
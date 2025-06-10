from typing import List
import yaml
from pydantic import BaseModel, ConfigDict


class NeuronConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    folder_name: str
    model_type: str
    n_node_types: int
    node_type_names: List[str]
    n_node_types_per_type: List[int]
    filter_length_max: int
    n_frames: int
    dt: float

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, 'r') as file:
            raw_config = yaml.safe_load(file)
        return NeuronConfig(**raw_config)

    def pretty(self):
        return yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False, indent=2)
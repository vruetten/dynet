from typing import Optional, Literal, Annotated, Dict
import yaml
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Union

# Sub-config schemas for ParticleGraph

class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    N_poisson: int = 4  # Number of Poisson nodes

    N_exponential: int =  20  # Number of exponential nodes
    N_exponential_groups: int =  3  # Number of groups of exponential nodes 

    N_oscillator: int = 10  # Number of oscillator nodes
    N_oscillator_groups: int =  1  # Number of groups of oscillator nodes

    N_gaussian_kernel: int =  0  # Number of gaussian kernel nodes
    N_gaussian_kernel_groups: int =  3  # Number of groups of gaussian kernel nodes

    N_damped_oscillation: int =  0  # Number of damped oscillation nodes
    N_damped_oscillation_groups: int =  3  # Number of groups of damped oscillation nodes

    N_bump: int = 0  # Number of bump nodes

    N_alpha: int = 20  # Number of alpha nodes
    N_alpha_groups: int = 3  # Number of groups of alpha nodes
    N_biexponential: int = 0

    kernel_duration: int = 3

    T: int = 400  # Simulation duration
    DT: float = 0.01  # Time step

# Main config schema for SimulationConfig

class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    description: Optional[str] = 'test'
    output_dir: str = 'none'
    simulation: SimulationConfig
    verbose: bool = True

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, 'r') as file:
            raw_config = yaml.safe_load(file)
        return SimulationConfig(**raw_config)

    def pretty(self):
        return yaml.dump(self, default_flow_style=False, sort_keys=False, indent=4)


if __name__ == '__main__':

    config_file = './config/test.yaml' # Insert path to config file
    config = SimulationConfig.from_yaml(config_file)
    print(config.pretty())

    print('Successfully loaded config file. Model description:', config.description)
    

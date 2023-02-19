import yaml
import os
from preprocess.configs.dotconfig import DotConfig

class Config(DotConfig):
    def __init__( self ):

        dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = f'{dir}/config.yaml'
        with open(config_file_path) as file:
            super().__init__(yaml.safe_load(file))
        

config = Config()

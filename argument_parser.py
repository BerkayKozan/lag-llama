import argparse
from constants import ARGUMENTS

class ArgumentParser:
    def __init__(self):
        self.args = self.parse_arguments()
    
    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        for arg in ARGUMENTS:
            flags = arg.pop("flags")
            arg_name = flags[-1].lstrip("-").replace("-", "_")
            parser.add_argument(*flags, **arg)
            
        return parser.parse_args()
import argparse
from constants import ARGUMENTS, DEFAULT_ARGS

class ArgumentParser:
    def __init__(self):
        self.args = self.parse_arguments()
    
    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        for arg in ARGUMENTS:
            flags = arg.pop("flags")
            arg_name = flags[-1].lstrip("-").replace("-", "_")
            # Use default from DEFAULT_ARGS if it exists
            if arg_name in DEFAULT_ARGS:
                arg['default'] = DEFAULT_ARGS[arg_name]
            parser.add_argument(*flags, **arg)
            
        return parser.parse_args()
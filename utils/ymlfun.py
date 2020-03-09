"""
Description: YAML reading class for handling configurations and parameters.
             You could overwrite some parameters when they are specified from CLI.
TODO:       * s
Reference:
"""
import argparse
import yaml
import collections

# Adapted from https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
def flatten_dict(var_dict, parent_key='', sep='_'):
    """
    A recursive function to flatten out the dictionary.
    :param var_dict: the nested dictionary you want to flatten
    :param parent_key: if specified, use that as the header for the new key
    :param sep: what kind of separator you want to use when joining dictionary elements
    :return dict(items): flattened dictionary
    """
    items = []
    for key, value in var_dict.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, collections.MutableMapping):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)

def get_parser(config_name="config.yaml"):
    """
    A function to create parser given default configurations from yaml.
    :param config_name: name of the configuration file
    :return parser: argparse parser
    """
    parser = argparse.ArgumentParser()
    # Read configuration file for defaults
    try:
        parser.add_argument('-c', '--config', type=argparse.FileType(mode='r'), default=config_name)
    except e:
        print(e)

    return parser

def get_args(parser):
    """
    A function to create args given default values from config file and overwrite if necessary.
    :param parser: argparse parser containing configuration file for default values
    :return args: arguments updated by command line inputs
    """
    args, _ = parser.parse_known_args()
    # Make sure the config exists
    if args.config:
        # Load the default configurations from the yaml file
        defaults = yaml.load(args.config)
        # Flatten the nested dictionary
        flat_defaults = flatten_dict(defaults)
        # Unroll what's inside the yaml
        opt_args = [['--' + key] for key, _ in flat_defaults.items()]
        opt_kwargs = [{'dest': key, 'type': type(value), 'default': value} for key, value in flat_defaults.items()]
        # Put the unrolled arguments into parser
        for p_args, p_kwargs in zip(opt_args, opt_kwargs):
            parser.add_argument(*p_args, **p_kwargs)
        args = parser.parse_args()

    return args


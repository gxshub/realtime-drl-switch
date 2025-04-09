import yaml
import json
import os


from rt_drl_safeguard.utils.randomization import SwitchConfigurationSampler


def load_config(file_path):
    """
    Loads a YAML or JSON file into a Python dictionary.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary representing the configuration data, or None if an error occurs.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    try:
        with open(file_path, 'r') as file:
            if file_path.lower().endswith(('.yaml', '.yml')):
                return yaml.safe_load(file)
            elif file_path.lower().endswith('.json'):
                return json.load(file)
            else:
                print("Error: Unsupported file format. Please use YAML or JSON.")
                return None

    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in {file_path}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def load_configs(directory_path):
    """
    Loads all YAML and JSON files from a directory into a list of dictionaries.

    Args:
        directory_path (str): The path to the directory containing configuration files.

    Returns:
        list: A list of dictionaries representing the configuration data, or None if an error occurs.
    """
    configs = []

    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        print(f"Error: Directory not found or is not a directory: {directory_path}")
        return None

    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            if os.path.isfile(file_path):
                if filename.lower().endswith(('.yaml', '.yml')):
                    with open(file_path, 'r') as file:
                        data = yaml.safe_load(file)
                        if data:
                            configs.append(data)
                elif filename.lower().endswith('.json'):
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        if data:
                            configs.append(data)

        return configs

    except (yaml.YAMLError, json.JSONDecodeError) as e:
        print(f"Error: Invalid file format in directory {directory_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def load_switch_random_params(file_path):
    raw_config_data = load_config(file_path)
    config_params = raw_config_data["params"]
    if len(config_params) == 0:
        raise ValueError("Switch configuration cannot be empty.")
    elif len(config_params) == 1:
        probabilities = [1]
    elif "probs" in raw_config_data:
            probabilities = raw_config_data["probs"]
    else:
        probabilities = [1.0 / len(config_params)] * len(config_params)
    return config_params, probabilities

def load_rule_based_ctrl_config(file_path, ctrl_name):
    raw_config_dict = load_config(file_path)
    config = raw_config_dict.get(ctrl_name, None)
    if config:
        return config
    else:
        raise ValueError("No configuration exists for {}".format(ctrl_name))

def load_rule_based_ctrl_configs(file_path):
    config_dict = load_config(file_path)
    return config_dict
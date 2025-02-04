import pandas as pd
import yaml
import os

def load_config(verbose=False):
    yml_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    config = yaml.safe_load(open(yml_path))

    if verbose:
        print(f"config dataset_name: {config['dataset_name']} with path: {config['dataset_path']}")

    return config

def init_dataset(config, verbose=False):
    supported_formats = ('mp3', 'wav', 'flac', 'm4a')
    dataset_path = config['dataset_path']
    list_files = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if verbose:
                print(f"Found file: {file}")
            if file.split('.')[-1] in supported_formats:
                list_files.append({'id': file.split('.')[0], 'path': os.path.join(root, file)})
                
    return pd.DataFrame(list_files)

def main():
    config = load_config(True)
    df = init_dataset(config, True)
    print(df.head())

if __name__ == '__main__':
    main()
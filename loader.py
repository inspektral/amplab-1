import pandas as pd
import numpy as np
import yaml
import os

def load_config(verbose=False):
    yml_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    config = yaml.safe_load(open(yml_path))

    if verbose:
        print(f"config dataset_name: {config['dataset_name']} with path: {config['dataset_path']}")

    return config

def init_dataset(config):
    dataset_path = config['dataset_path']
    df = pd.DataFrame()

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            print(file)
            if file.endswith('.mp3'):
                df = pd.concat([df, pd.DataFrame({'file': file.split('.')[:-1], 'path': [os.path.join(root, file)]})])
                
    
    return df

def main():
    config = load_config(True)
    df = init_dataset(config)
    print(df.head())

if __name__ == '__main__':
    main()
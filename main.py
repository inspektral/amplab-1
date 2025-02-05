import loader
from analyzer import Analyzer
import json
import pandas as pd
import tqdm

NEW_DATASET = False
ANALYSIS_RESULTS = 'results.csv'

def main():
    if NEW_DATASET:
        config = loader.load_config(True)
        df = loader.init_dataset(config, True)
    else:
        df = pd.read_csv(ANALYSIS_RESULTS)

    analyzer = Analyzer()
    
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Analyzing files"):

        if 'analyzed' in df.columns and row['analyzed'] == True:
            print(f"File already analyzed: {row['path']}")
        
        else:
            print(f"Analyzing file: {row['path']}")
            results = analyzer.analyze(row['path'])
            
            for key in results:
                if isinstance(results[key], list):
                    df.loc[index, key] = json.dumps(results[key])
                else:
                    df.loc[index, key] = results[key]


            df.to_csv(ANALYSIS_RESULTS, index=False)

if __name__ == '__main__':
    main()
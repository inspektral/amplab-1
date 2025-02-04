import loader
from analyzer import Analyzer
import json

def main():
    config = loader.load_config(True)
    df = loader.init_dataset(config, True)
    
    analyzer = Analyzer()
    
    for index, row in df.iterrows():
        print(f"Analyzing file: {row['path']}")
        results = analyzer.analyze(row['path'])
        
        print(f"Results: {results}")
        for key in results:
            if isinstance(results[key], list):
                df.loc[index, key] = json.dumps(results[key])
            else:
                df.loc[index, key] = results[key]


        if index > 10:
            break

    df.to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()
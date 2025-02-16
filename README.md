# amplab-1

A Python-based audio analysis tool that processes audio files using various machine learning models to extract musical features.

## Features

Analyzes audio files for multiple characteristics including:
- Danceability
- Genre classification
- Voice/instrumental detection
- Emotional content analysis
- Processes files in batches
- Saves analysis results to CSV format
- Supports incremental processing with checkpoint saving

## Setup

1. Install dependencies
2. Ensure model weights are present in the `weights` directory:
    - `danceability-discogs-effnet-1.pb`
    - `emomusic-msd-musicnn-2.pb`
    - `genre_discogs400-discogs-effnet-1.pb`
    - `voice_instrumental-discogs-effnet-1.pb`

## Usage

To analyze audio files:

The script will:
1. Load configuration from `config.yml`
2. Process audio files in the `dataset_path` directory
3. Save results to `results.csv`

## Project Structure

- `analyzer.py` - Core analysis functionality
- `loader.py` - Data and configuration loading utilities
- `main.py` - Main execution script
- `model_metadata` - Model configuration files
- `weights` - Pre-trained model weights
- `results.csv` - Analysis output file


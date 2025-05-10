# Mars Dataset Uploader

A tool for uploading Mars image datasets to Hugging Face in three formats:
- Classification
- Object Detection
- Segmentation

## Installation

### Prerequisites

- Python 3.11 or newer
- [Poetry](https://python-poetry.org/docs/#installation)
- XZ Libraries (required for LZMA compression)

```bash
# Install XZ library on macOS (for LZMA support)
brew install xz

# Make sure Python is compiled with LZMA support
# If using pyenv:
CFLAGS="-I$(brew --prefix xz)/include" LDFLAGS="-L$(brew --prefix xz)/lib" pyenv install --force 3.12.9
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mars_dataset_uploader.git
cd mars_dataset_uploader
```

2. Install dependencies with Poetry:
```bash
poetry env use $(which python)  # Use your system Python
poetry install
```

3. Activate the virtual environment:
```bash
# Poetry 2.0+
source $(poetry env info --path)/bin/activate
```

## Usage

### 1. Classification Dataset Upload

Upload a classification dataset to Hugging Face:

```bash
python -m classification_pipeline.cli \
  --data_dir /path/to/classification/data \
  --dataset_name your-username/dataset-name \
  --token your_huggingface_token \
  [--annotation_file /path/to/annotation.csv] \
  [--class_names class1 class2 class3 ...] \
  [--few_shot_dir /path/to/few_shot] \
  [--partition_dir /path/to/partitions] \
  [--mapping_file /path/to/mapping.json] \
  [--private]
```

#### Required Arguments:
- `--data_dir`: Path to the directory containing images
- `--dataset_name`: Name for the dataset on Hugging Face (username/dataset-name)
- `--token`: Hugging Face API token

#### Optional Arguments:
- `--annotation_file`: Path to CSV file with annotations
- `--class_names`: List of class names
- `--num_classes`: Number of classes (if class_names not provided)
- `--few_shot_dir`: Directory containing few-shot JSON files
- `--partition_dir`: Directory containing partition JSON files
- `--mapping_file`: Path to JSON file mapping class IDs to names
- `--private`: Make the dataset private
- `--debug`: Enable debug mode with additional logging

### 2. Multi-Label Classification Dataset Upload

Upload a multi-label classification dataset to Hugging Face, where images can belong to multiple classes simultaneously:

```bash
poetry run python -m multi_label_classification_pipeline.cli \
  --dataset-name your-username/dataset-name \
  --data-dir /path/to/multi_label/data \
  --annotation-file /path/to/annotation.csv \
  --mapping-file /path/to/mapping.json \
  --few-shot-dir /path/to/few_shot \
  --token your_huggingface_token \
  [--private] \
  [--text-column file_id] \
  [--label-column label] \
  [--feature-name-column feature_name] \
  [--split-column split]
```

#### Required Arguments:
- `--dataset-name`: Name for the dataset on Hugging Face (username/dataset-name)
- `--data-dir`: Path to the directory containing images
- `--annotation-file`: Path to CSV file with annotations
- `--token`: Hugging Face API token

#### Optional Arguments:
- `--mapping-file`: Path to JSON file mapping class IDs to names
- `--few-shot-dir`: Directory containing few-shot CSV files
- `--private`: Make the dataset private
- `--text-column`: Name of the column containing image filenames (default: "file_id")
- `--label-column`: Name of the column containing labels (default: "label")
- `--feature-name-column`: Name of the column containing feature names (default: "feature_name")
- `--split-column`: Name of the column containing split information (default: "split")

### 3. Object Detection Dataset Upload

Upload an object detection dataset (COCO or YOLO format) to Hugging Face:

```bash
python -m object_detection_pipeline.object_detection_uploader \
  --data_dir /path/to/detection/data \
  --dataset_name your-username/dataset-name \
  --token your_huggingface_token \
  --format {coco,yolo} \
  [--config_file /path/to/config.yaml] \
  [--include_pascal_voc] \
  [--include_coco] \
  [--include_yolo] \
  [--private]
```

#### Required Arguments:
- `--data_dir`: Path to data directory containing images and annotations
- `--dataset_name`: Name for dataset on Hugging Face (username/dataset-name)

#### Optional Arguments:
- `--token`: Hugging Face API token (will use stored token if not provided)
- `--format`: Format of annotations (`coco` or `yolo`, default: `coco`)
- `--config_file`: Path to configuration file for YOLO format
- `--private`: Make dataset private
- `--include_pascal_voc`: Include Pascal VOC format annotations as a column
- `--include_coco`: Include COCO format annotations as a column
- `--include_yolo`: Include YOLO format annotations as a column

### 4. Segmentation Dataset Upload

Upload a segmentation dataset to Hugging Face:

```bash
python -m segmentation_dataset_uploader.segmentation_uploader \
  --data_dir /path/to/segmentation/data \
  --dataset_name your-username/dataset-name \
  --token your_huggingface_token \
  [--config_file /path/to/config.yaml] \
  [--mapping_file /path/to/mapping.json] \
  [--private]
```

#### Required Arguments:
- `--data_dir`: Directory containing the data
- `--dataset_name`: Name for the dataset on HF (username/dataset-name)

#### Optional Arguments:
- `--token`: Hugging Face API token
- `--config_file`: Path to config file (optional)
- `--mapping_file`: Path to mapping.json file (optional)
- `--private`: Make the dataset private

## Examples

### Classification Example

```bash
python -m classification_pipeline.cli \
  --annotation_file /path/to/DoMars16K/annotation.csv \
  --data_dir /path/to/DoMars16K/data \
  --dataset_name username/domarks16k \
  --token hf_your_token_here \
  --few_shot_dir /path/to/DoMars16K/few_shot \
  --partition_dir /path/to/DoMars16K/partitions \
  --class_names aec ael cli cra fse fsf fsg fss mix rid rou sfe sfx smo tex \
  --mapping_file /path/to/DoMars16K/mapping.json
```

### Multi-Label Classification Example

```bash
poetry run python -m multi_label_classification_pipeline.cli \
  --dataset-name username/mars-multi-label-classification \
  --data-dir multi_label_mer/data \
  --annotation-file multi_label_mer/annotation.csv \
  --mapping-file multi_label_mer/mapping.json \
  --few-shot-dir multi_label_mer/few_shot \
  --token hf_your_token_here
```

### Object Detection Example

```bash
python -m object_detection_pipeline.object_detection_uploader \
  --data_dir /path/to/mars_objects/coco_format \
  --dataset_name username/mars-objects-detection \
  --token hf_your_token_here \
  --format coco \
  --private
```

### Segmentation Example

```bash
python -m segmentation_dataset_uploader.segmentation_uploader \
  --data_dir /path/to/mars_segmentation \
  --dataset_name username/mars-segmentation \
  --token hf_your_token_here \
  --mapping_file /path/to/mapping.json
```

## Troubleshooting

### Missing _lzma Module Error

If you encounter `ModuleNotFoundError: No module named '_lzma'`:

1. Install XZ library:
   ```bash
   brew install xz  # macOS
   sudo apt-get install liblzma-dev  # Ubuntu/Debian
   ```

2. Reinstall Python with LZMA support:
   ```bash
   # If using pyenv
   CFLAGS="-I$(brew --prefix xz)/include" LDFLAGS="-L$(brew --prefix xz)/lib" pyenv install --force 3.12.9
   ```

3. Recreate Poetry environment:
   ```bash
   poetry env remove existing-env-name
   poetry env use $(pyenv which python)
   poetry install
   ```
"""
Core functionality for uploading Mars datasets to Hugging Face.
"""

import os
import json
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime

import pandas as pd
from datasets import Dataset, DatasetDict, Image, ClassLabel, Features, DatasetInfo
from tqdm import tqdm
import huggingface_hub
import glob

# Default dataset information
DEFAULT_DATASET_INFO = {
    "description": "Mars Image Classification Dataset",
    "license": "cc-by-4.0",
    "tags": ["mars", "image-classification"],
    "version": "1.0"
}

def create_dataset_info(
    base_info: Dict[str, Any],
    class_names: Optional[List[str]] = None,
    few_shot_config: Optional[Dict] = None
) -> DatasetInfo:
    """
    Create a DatasetInfo object with metadata.
    
    Args:
        base_info: Base dataset information
        class_names: List of class names
        few_shot_config: Few-shot configuration dictionary
    
    Returns:
        DatasetInfo object with metadata
    """
    info = DatasetInfo(description=base_info.get("description", ""))
    
    # Add basic metadata
    info.license = base_info.get("license", "")
    info.tags = base_info.get("tags", [])
    
    # Add class names if available
    if class_names:
        info.features = Features({
            'image': Image(),
            'label': ClassLabel(names=class_names)
        })
    
    # Add few-shot configuration if available
    if few_shot_config:
        # Ensure metadata dictionary exists
        if not hasattr(info, 'metadata') or info.metadata is None:
            info.metadata = {}
        info.metadata['few_shot_config'] = json.dumps(few_shot_config)
    
    return info


def create_dataset_dict(
    annotation_file: str,
    data_dir: str,
    text_column: str = "file_id",
    label_column: str = "label",
    split_column: str = "split",
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    few_shot_files: Optional[List[str]] = None,
    partition_files: Optional[List[str]] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
    mapping_file: Optional[str] = None
) -> Tuple[DatasetDict, List[Dict], List[Dict], str, Optional[str]]:
    """
    Create a DatasetDict from the Mars dataset annotation file and image directory.
    If few_shot_file is provided, include few-shot splits for training data only.
    If partition_file is provided, include splits based on partition definitions.
    
    Args:
        annotation_file: Path to the annotation CSV file
        data_dir: Path to the data directory containing images
        text_column: Name of the image path column (defaults to 'file_id' for new format)
        label_column: Name of the label column
        split_column: Name of the split column
        class_names: List of class names (e.g., ['crater', 'dust_devil', 'rock'])
        num_classes: Number of classes (if class_names not provided)
        few_shot_files: List of paths to few-shot CSV files (optional)
        partition_files: List of paths to partition JSON files (optional)
        dataset_info: Optional dictionary containing dataset metadata
        mapping_file: Optional path to mapping file
    
    Returns:
        Tuple of (DatasetDict containing the datasets, 
                 List of few-shot configurations, 
                 List of partition configurations,
                 annotation file path,
                 mapping file path)
    """
    print(f"Attempting to read annotation file: {annotation_file}")
    
    all_few_shot_configs = []
    all_partition_configs = []
    dataset_dict = {}
    
    # First try to load mapping file if available to get class names
    if mapping_file and os.path.exists(mapping_file):
        try:
            print(f"Loading class names from mapping file: {mapping_file}")
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
                print(f"Mapping file contents: {mapping_data}")
                
            # If mapping has integer keys, extract class names
            if all(k.isdigit() for k in mapping_data.keys()):
                max_class_id = max(int(idx) for idx in mapping_data.keys())
                extracted_class_names = ["" for _ in range(max_class_id + 1)]
                for idx, name in mapping_data.items():
                    extracted_class_names[int(idx)] = name
                
                # Only use if we don't have class names already
                if class_names is None:
                    class_names = extracted_class_names
                    print(f"Loaded {len(class_names)} classes from mapping file: {class_names}")
        except Exception as e:
            print(f"Error loading mapping file: {e}")
    
    # Check if annotation file exists
    if not os.path.exists(annotation_file):
        print(f"Warning: Annotation file '{annotation_file}' not found. Attempting to infer dataset structure from directory.")
        
        # Try to infer from directory structure
        try:
            # Detect the dataset structure
            print("\nAnalyzing data directory structure...")
            # Find all directories at depth 1 (these could be splits)
            potential_splits = []
            potential_classes = []
            
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    if item in ['train', 'val', 'test']:
                        potential_splits.append(item)
                    else:
                        potential_classes.append(item)
            
            print(f"Potential splits found: {potential_splits}")
            if potential_classes:
                print(f"Potential class directories found: {potential_classes}")
            
            # Determine the structure: 
            # 1. Standard: data_dir/[train,val,test]/[class1,class2,...]/images
            # 2. DoMars: data_dir/[train,val,test]/[class_folders]
            # 3. Flat: data_dir/[class1,class2,...]/images
            
            # Auto-detect dataset structure if annotation file is missing
            features = Features({
                'image': Image(),
                'label': ClassLabel(names=class_names) if class_names else None
            })
            
            # First try standard structure with split directories
            if potential_splits:
                print("Detected potential split directories. Checking for standard structure...")
                
                # Check if classes are already extracted from mapping file
                if class_names is None:
                    # Try to get class names from subdirs of a split directory
                    for split in potential_splits:
                        split_dir = os.path.join(data_dir, split)
                        split_subdirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
                        if split_subdirs:
                            print(f"Found potential class directories in {split}: {split_subdirs}")
                            class_names = split_subdirs
                            features = Features({
                                'image': Image(),
                                'label': ClassLabel(names=class_names)
                            })
                            break
                
                # Now create datasets for each split
                for split in potential_splits:
                    split_dir = os.path.join(data_dir, split)
                    print(f"Processing {split} directory: {split_dir}")
                    
                    # If we have class names, look for images in each class dir
                    if class_names:
                        examples = []
                        
                        for class_idx, class_name in enumerate(class_names):
                            class_dir = os.path.join(split_dir, class_name)
                            if not os.path.isdir(class_dir):
                                print(f"No directory found for class '{class_name}' in {split}")
                                continue
                                
                            # Look for images directly in class dir
                            image_files = []
                            for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                                image_files.extend(glob.glob(os.path.join(class_dir, f'*{ext}')))
                            
                            if not image_files:
                                # Try looking for images in an 'images' subdirectory
                                images_dir = os.path.join(class_dir, 'images')
                                if os.path.isdir(images_dir):
                                    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                                        image_files.extend(glob.glob(os.path.join(images_dir, f'*{ext}')))
                            
                            if image_files:
                                print(f"Found {len(image_files)} images for class '{class_name}' in {split}")
                                examples.extend([
                                    {
                                        'image': img_path,
                                        'label': class_idx
                                    }
                                    for img_path in image_files
                                ])
                            else:
                                print(f"No images found for class '{class_name}' in {split}")
                        
                        if examples:
                            dataset_dict[split] = Dataset.from_list(examples, features=features)
                            print(f"Created {split} dataset with {len(examples)} examples")
                    else:
                        # For DoMars, images are directly in the split directory
                        # with no class subdirectories. In this case, we use the mapping file
                        print("No class directories found. Looking for images directly in split directory...")
                        
                        examples = []
                        image_files = []
                        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                            image_files.extend(glob.glob(os.path.join(split_dir, f'*{ext}')))
                            
                        if image_files:
                            # We'll need to infer class from filename or use a placeholder
                            print(f"Found {len(image_files)} images directly in {split} directory")
                            # This needs a separate mapping to work properly
                            # For now, just assign a default label of 0
                            examples = [
                                {
                                    'image': img_path,
                                    'label': 0  # Placeholder label
                                }
                                for img_path in image_files
                            ]
                            dataset_dict[split] = Dataset.from_list(examples)
                            print(f"Created {split} dataset with {len(examples)} examples (all assigned default class 0)")
            
            # If we didn't find any splits or couldn't create datasets, try a flat structure
            if not dataset_dict and potential_classes:
                print("\nNo standard structure detected. Trying flat structure (no split directories)...")
                
                # Use potential_classes as class names if we don't have them
                if class_names is None:
                    class_names = potential_classes
                    features = Features({
                        'image': Image(),
                        'label': ClassLabel(names=class_names)
                    })
                
                # Create a single dataset with all images
                examples = []
                for class_idx, class_name in enumerate(class_names):
                    if class_name in potential_classes:
                        class_dir = os.path.join(data_dir, class_name)
                        
                        # Look for images directly in class dir
                        image_files = []
                        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                            image_files.extend(glob.glob(os.path.join(class_dir, f'*{ext}')))
                        
                        if image_files:
                            print(f"Found {len(image_files)} images for class '{class_name}'")
                            examples.extend([
                                {
                                    'image': img_path,
                                    'label': class_idx
                                }
                                for img_path in image_files
                            ])
                
                if examples:
                    # Since we don't have explicit splits, put everything in train
                    dataset_dict['train'] = Dataset.from_list(examples, features=features)
                    print(f"Created dataset with {len(examples)} examples (all in 'train' split)")
            
            # If the above standard approaches didn't work, try DoMars16K specific structure
            # DoMars16K has: data_dir/[train,val,test]/[aec,ael,cli,...] where the latter are classes
            if not dataset_dict and potential_splits:
                print("\nTrying DoMars16K specific structure...")
                
                for split in potential_splits:
                    split_dir = os.path.join(data_dir, split)
                    
                    # Get all subdirectories in the split dir - these should be the classes
                    class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
                    
                    if class_dirs:
                        print(f"Found potential class directories in {split}: {class_dirs}")
                        
                        # If we don't have class names, use the directories as class names
                        if class_names is None:
                            class_names = class_dirs
                            features = Features({
                                'image': Image(),
                                'label': ClassLabel(names=class_names)
                            })
                        
                        examples = []
                        for class_idx, class_name in enumerate(class_names):
                            if class_name in class_dirs:
                                class_dir = os.path.join(split_dir, class_name)
                                
                                # Look for images directly in the class directory
                                image_files = []
                                for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                                    image_files.extend(glob.glob(os.path.join(class_dir, f'*{ext}')))
                                
                                if image_files:
                                    print(f"Found {len(image_files)} images for class '{class_name}' in {split}")
                                    examples.extend([
                                        {
                                            'image': img_path,
                                            'label': class_idx
                                        }
                                        for img_path in image_files
                                    ])
                        
                        if examples:
                            dataset_dict[split] = Dataset.from_list(examples, features=features)
                            print(f"Created {split} dataset with {len(examples)} examples")
        
        except Exception as e:
            print(f"Error inferring dataset structure: {e}")
            import traceback
            traceback.print_exc()
            if not dataset_dict:
                raise ValueError(f"Could not create dataset: Annotation file '{annotation_file}' not found and could not infer dataset structure")
            
    else:
        # Normal flow with annotation file
        try:
            df = pd.read_csv(annotation_file)
            
            if class_names is None:
                if num_classes is None:
                    num_classes = df[label_column].nunique()
                class_names = [f"class_{i}" for i in range(num_classes)]
            
            class_label = ClassLabel(num_classes=len(class_names), names=class_names)
            features = Features({
                'image': Image(),
                'label': class_label
            })
            
            # Create standard splits
            for split in ['train', 'test', 'val']:
                split_df = df[df[split_column] == split].copy()
                if not split_df.empty:
                    # Create the image column with full paths and prepare data dictionary
                    # We need to convert numeric label to class name
                    data_dict = {}
                    
                    # Get image paths using class names instead of numeric labels
                    image_paths = []
                    for _, row in split_df.iterrows():
                        label_val = row[label_column]
                        # Check if we have class names to map this label
                        if class_names and 0 <= label_val < len(class_names):
                            # Use the class name from the mapping instead of the numeric label
                            class_name = class_names[label_val]
                            image_path = os.path.join(data_dir, split, class_name, row[text_column])
                        else:
                            # Fallback to using the label value directly
                            image_path = os.path.join(data_dir, split, str(label_val), row[text_column])
                        
                        # Verify that the image exists
                        if not os.path.exists(image_path):
                            print(f"Warning: Image not found: {image_path}")
                            # Try alternate path without class subdirectory
                            alt_path = os.path.join(data_dir, split, row[text_column])
                            if os.path.exists(alt_path):
                                print(f"Found image at alternate path: {alt_path}")
                                image_path = alt_path
                            else:
                                # If image doesn't exist, we might want to skip it
                                continue
                        
                        image_paths.append(image_path)
                    
                    # Create data dictionary with images that exist
                    if image_paths:
                        data_dict = {
                            'image': image_paths,
                            'label': [split_df.iloc[i][label_column] for i in range(len(image_paths))]
                        }
                        dataset_dict[split] = Dataset.from_dict(
                            data_dict,
                            features=features
                        )
                    else:
                        print(f"Warning: No valid images found for split {split}")
        except Exception as e:
            print(f"Error reading annotation file: {e}")
            import traceback
            traceback.print_exc()
            if not dataset_dict:
                raise ValueError(f"Could not create dataset from annotation file: {str(e)}")

    # Add few-shot splits if provided
    if few_shot_files:
        for few_shot_file in few_shot_files:
            if os.path.exists(few_shot_file):
                try:
                    # Extract file prefix for split name (e.g., "10_shot" from "10_shot.csv" or "10_shot.json")
                    file_prefix = os.path.splitext(os.path.basename(few_shot_file))[0]
                    few_shot_split_name = f"few_shot_train_{file_prefix}"
                    
                    # Check if this is a CSV file (new format) or JSON file (old format)
                    if few_shot_file.endswith('.csv'):
                        # New format - CSV file similar to annotation file
                        few_shot_df = pd.read_csv(few_shot_file)
                        print(f"Loaded few-shot file {few_shot_file} with {len(few_shot_df)} rows")
                        
                        # Filter for train split
                        train_df = few_shot_df[few_shot_df[split_column] == 'train'].copy()
                        if not train_df.empty:
                            # Store config with filename info for tracking
                            config_info = {"filename": os.path.basename(few_shot_file), "config": {"train": f"{len(train_df)} examples"}}
                            all_few_shot_configs.append(config_info)
                            
                            # Create dataset using the same path construction as for the main dataset
                            image_paths = []
                            labels = []
                            
                            for _, row in train_df.iterrows():
                                label_val = row[label_column]
                                # Check if we have class names to map this label
                                if class_names and 0 <= label_val < len(class_names):
                                    # Use the class name from the mapping instead of the numeric label
                                    class_name = class_names[label_val]
                                    image_path = os.path.join(data_dir, 'train', class_name, row[text_column])
                                else:
                                    # Fallback to using the label value directly
                                    image_path = os.path.join(data_dir, 'train', str(label_val), row[text_column])
                                
                                # Verify that the image exists
                                if not os.path.exists(image_path):
                                    # Try alternate path without class subdirectory
                                    alt_path = os.path.join(data_dir, 'train', row[text_column])
                                    if os.path.exists(alt_path):
                                        image_path = alt_path
                                    else:
                                        # Skip missing images
                                        continue
                                
                                image_paths.append(image_path)
                                labels.append(label_val)
                            
                            if image_paths:
                                data_dict = {
                                    'image': image_paths,
                                    'label': labels
                                }
                                dataset_dict[few_shot_split_name] = Dataset.from_dict(
                                    data_dict,
                                    features=features
                                )
                                print(f"Created {few_shot_split_name} dataset with {len(image_paths)} examples")
                            else:
                                print(f"Warning: No valid images found for few_shot file {few_shot_file}")
                    else:
                        # Legacy format - JSON file with {"train": {"class": [img, ...]}}
                        with open(few_shot_file, 'r') as f:
                            few_shot_data = json.load(f)
                            # Store config with filename info for tracking
                            config_info = {"filename": os.path.basename(few_shot_file), "config": few_shot_data}
                            all_few_shot_configs.append(config_info)
                            
                            # Create few-shot dataset only for training data
                            if 'train' in few_shot_data: # Assuming structure {"train": {"class": [img, ...]}}
                                examples = []
                                for class_idx, class_name in enumerate(class_names):
                                    if class_name in few_shot_data['train']:
                                        examples.extend([
                                            {
                                                'image': os.path.join(data_dir, 'train', class_name, image_name),
                                                'label': class_idx
                                            }
                                            for image_name in few_shot_data['train'][class_name]
                                            # Add existence check?
                                            # if os.path.exists(os.path.join(data_dir, 'train', class_name, image_name))
                                        ])
                                
                                if examples:
                                    dataset_dict[few_shot_split_name] = Dataset.from_list(
                                        examples,
                                        features=features
                                    )
                                else:
                                    print(f"Warning: No valid training examples found for few-shot file: {few_shot_file}")
                            else:
                                print(f"Warning: 'train' key not found in few-shot file: {few_shot_file}")
                except Exception as e:
                    print(f"Error processing few-shot file {few_shot_file}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: Few-shot file path not found: {few_shot_file}")

    # Add partition splits if provided
    if partition_files and 'train' in dataset_dict:
        # Use the basename info approach only if we have a DataFrame
        try:
            if 'df' in locals():
                # Pre-calculate basename info once
                df['basename'] = df[text_column].apply(os.path.basename)
                basename_to_info = {row['basename']: {'relative_path': row[text_column], 'split': row[split_column], 'label': row[label_column]} 
                                for _, row in df.iterrows()}
        except Exception as e:
            print(f"Warning: Could not prepare basename info for partitions: {e}")
        
        for partition_file in partition_files:
            if os.path.exists(partition_file):
                try:
                    # Extract file prefix (e.g., "0.02x" from "0.02x_partition.json")
                    # Improved prefix extraction to handle potential variations
                    base = os.path.basename(partition_file)
                    if base.endswith('_partition.json'):
                        file_prefix = base[:-len('_partition.json')]
                    elif base.endswith('.json'):
                        file_prefix = base[:-len('.json')]
                    elif base.endswith('.csv'):
                        file_prefix = base[:-len('.csv')]
                    else:
                        file_prefix = base # Fallback
                    partition_split_name = f"partition_train_{file_prefix}"
                    
                    # Check if this is a CSV file (new format) or JSON file (old format)
                    if partition_file.endswith('.csv'):
                        # New format - CSV file similar to annotation file
                        partition_df = pd.read_csv(partition_file)
                        print(f"Loaded partition file {partition_file} with {len(partition_df)} rows")
                        
                        # Filter for train split
                        train_df = partition_df[partition_df[split_column] == 'train'].copy()
                        if not train_df.empty:
                            # Store config with filename info for tracking
                            config_info = {"filename": os.path.basename(partition_file), "config": {"train": f"{len(train_df)} examples"}}
                            all_partition_configs.append(config_info)
                            
                            # Create dataset using the same path construction as for the main dataset
                            image_paths = []
                            labels = []
                            
                            for _, row in train_df.iterrows():
                                label_val = row[label_column]
                                # Check if we have class names to map this label
                                if class_names and 0 <= label_val < len(class_names):
                                    # Use the class name from the mapping instead of the numeric label
                                    class_name = class_names[label_val]
                                    image_path = os.path.join(data_dir, 'train', class_name, row[text_column])
                                else:
                                    # Fallback to using the label value directly
                                    image_path = os.path.join(data_dir, 'train', str(label_val), row[text_column])
                                
                                # Verify that the image exists
                                if not os.path.exists(image_path):
                                    # Try alternate path without class subdirectory
                                    alt_path = os.path.join(data_dir, 'train', row[text_column])
                                    if os.path.exists(alt_path):
                                        image_path = alt_path
                                    else:
                                        # Skip missing images
                                        continue
                                
                                image_paths.append(image_path)
                                labels.append(label_val)
                            
                            if image_paths:
                                data_dict = {
                                    'image': image_paths,
                                    'label': labels
                                }
                                dataset_dict[partition_split_name] = Dataset.from_dict(
                                    data_dict,
                                    features=features
                                )
                                print(f"Created {partition_split_name} dataset with {len(image_paths)} examples")
                            else:
                                print(f"Warning: No valid images found for partition file {partition_file}")
                    else:
                        # Legacy format - JSON file with {"train": {"class": [img, ...]}}
                        with open(partition_file, 'r') as f:
                            try:
                                partition_data = json.load(f)
                                # Store config with filename info
                                config_info = {"filename": os.path.basename(partition_file), "config": partition_data}
                                all_partition_configs.append(config_info)

                                partition_train_examples = []
                                
                                # Use similar approach to few_shot files for simplicity and consistency
                                if 'train' in partition_data and isinstance(partition_data['train'], dict):
                                    for class_idx, class_name in enumerate(class_names):
                                        if class_name in partition_data['train'] and isinstance(partition_data['train'][class_name], list):
                                            partition_train_examples.extend([
                                                {
                                                    'image': os.path.join(data_dir, 'train', class_name, image_name),
                                                    'label': class_idx
                                                }
                                                for image_name in partition_data['train'][class_name]
                                                if os.path.exists(os.path.join(data_dir, 'train', class_name, image_name))
                                            ])
                                else:
                                    print(f"Warning: Key 'train' not found or its value is not a dictionary in {partition_file}. Cannot extract basenames for partition.")

                                if partition_train_examples:
                                    dataset_dict[partition_split_name] = Dataset.from_list(
                                        partition_train_examples,
                                        features=features
                                    )
                                else:
                                    print(f"Warning: No valid train images found for partition defined in {partition_file}")

                            except json.JSONDecodeError:
                                print(f"Warning: Could not decode JSON from partition file: {partition_file}")
                except Exception as e:
                    print(f"Error processing partition file {partition_file}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: Partition file path not found: {partition_file}")

    if not dataset_dict:
        raise ValueError("No valid data found in any split. Please check your directory structure and annotation file.")

    return DatasetDict(dataset_dict), all_few_shot_configs, all_partition_configs, annotation_file, mapping_file


def create_readme(
    dataset_name: str,
    dataset_dict: DatasetDict,
    class_names: Optional[List[str]] = None,
    few_shot_configs: Optional[List[Dict]] = None,
    partition_configs: Optional[List[Dict]] = None
) -> str:
    """
    Create README content for the dataset with YAML frontmatter.
    
    Args:
        dataset_name: Name of the dataset
        dataset_dict: DatasetDict to extract statistics from
        class_names: List of class names
        few_shot_configs: List of few-shot configuration dictionaries
        partition_configs: List of partition configuration dictionaries

    Returns:
        String containing the README content
    """
    # Get current date for metadata
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Get the basename of the dataset (the part after the last /)
    dataset_basename = dataset_name.split('/')[-1]
    
    # Create YAML frontmatter for the README with metadata
    readme = "---\n"
    readme += "annotations_creators:\n- expert-generated\n"
    readme += "language_creators:\n- found\n"
    readme += "language:\n- en\n"
    readme += "license:\n- cc-by-4.0\n"
    readme += "multilinguality:\n- monolingual\n"
    readme += "size_categories:\n- 10K<n<100K\n"
    readme += "source_datasets:\n- original\n"
    readme += "task_categories:\n- image-classification\n"
    readme += "task_ids:\n- multi-class-image-classification\n"
    readme += f"pretty_name: {dataset_basename}\n"
    
    # Add class label names dictionary if available
    if class_names:
        class_label_names = {str(i): name for i, name in enumerate(class_names)}
        readme += "dataset_info:\n"
        readme += "  features:\n"
        readme += "  - name: image\n"
        readme += "    dtype: image\n"
        readme += "  - name: label\n"
        readme += "    dtype:\n"
        readme += "      class_label:\n"
        readme += "        names:\n"
        
        # Add class names
        for class_id, class_name in class_label_names.items():
            readme += f"          '{class_id}': {class_name}\n"
    
    # End frontmatter
    readme += "---\n\n"
    
    # Add dataset title
    readme += f"# {dataset_basename}\n\n"
    
    # Add dataset description
    readme += "A Mars image classification dataset for planetary science research.\n\n"
    
    # Add metadata section
    readme += "## Dataset Metadata\n\n"
    
    # Add license, version, datePublished, and citeAs
    readme += "* **License:** CC-BY-4.0 (Creative Commons Attribution 4.0 International)\n"
    readme += "* **Version:** 1.0\n"
    readme += f"* **Date Published:** {current_date}\n"
    readme += "* **Cite As:** TBD\n\n"
    
    # Add class information
    readme += "## Classes\n\n"
    if class_names:
        readme += "This dataset contains the following classes:\n\n"
        class_list = "\n".join([f"- {i}: {name}" for i, name in enumerate(class_names)])
        readme += f"{class_list}\n\n"
    
    # Add split statistics
    readme += "## Statistics\n\n"
    for split_name, split_dataset in dataset_dict.items():
        readme += f"- **{split_name}**: {len(split_dataset)} images\n"
    readme += "\n"
    
    # Add Few-shot splits information only if relevant
    few_shot_splits = [split for split in dataset_dict.keys() if 'few_shot_train_' in split]
    if few_shot_splits:
        readme += "## Few-shot Splits\n\n"
        readme += "This dataset includes the following few-shot training splits:\n\n"
        for split in few_shot_splits:
            readme += f"- **{split}**: {len(dataset_dict[split])} images\n"
        readme += "\n"
        
        if few_shot_configs:
            readme += "Few-shot configurations:\n\n"
            for config in few_shot_configs:
                filename = config.get("filename", "unknown")
                readme += f"- **{filename}**\n"
                
    # Add Partition splits information only if relevant
    partition_splits = [split for split in dataset_dict.keys() if 'partition_train_' in split]
    if partition_splits:
        readme += "## Partition Splits\n\n"
        readme += "This dataset includes the following training data partitions:\n\n"
        for split in partition_splits:
            readme += f"- **{split}**: {len(dataset_dict[split])} images\n"
        readme += "\n"
    
    # Add usage example
    readme += "## Usage\n\n"
    readme += "```python\n"
    readme += "from datasets import load_dataset\n\n"
    readme += f"dataset = load_dataset(\"{dataset_name}\")\n"
    readme += "```\n\n"
    
    # Add information about the format
    readme += "## Format\n\n"
    readme += "Each example in the dataset has the following format:\n\n"
    readme += "```\n"
    readme += "{\n"
    readme += "  'image': Image(...),  # PIL image\n"
    readme += "  'label': int,         # Class label\n"
    readme += "}\n"
    readme += "```\n"
    
    return readme

def upload_to_huggingface(
    dataset_dict: DatasetDict,
    dataset_name: str,
    token: str,
    private: bool = False,
    dataset_info: Optional[Dict[str, Any]] = None,
    class_names: Optional[List[str]] = None,
    few_shot_configs: Optional[List[Dict]] = None,
    partition_configs: Optional[List[Dict]] = None,
    annotation_file: Optional[str] = None,
    mapping_file: Optional[str] = None
) -> None:
    """
    Upload the dataset to Hugging Face.
    
    Args:
        dataset_dict: DatasetDict to upload
        dataset_name: Name for dataset on Hugging Face
        token: Hugging Face API token
        private: Whether to make the dataset private
        dataset_info: Optional dictionary containing dataset metadata
        class_names: List of class names
        few_shot_configs: List of few-shot configuration dictionaries
        partition_configs: List of partition configuration dictionaries
        annotation_file: Path to annotation file to upload
        mapping_file: Path to mapping file to upload
    """
    from huggingface_hub import HfApi
    
    # Create API object
    api = HfApi(token=token)
    
    # First check if dataset exists - if not, create it
    try:
        # Try to get repository info
        try:
            api.repo_info(repo_id=dataset_name, repo_type="dataset")
            print(f"Dataset repository {dataset_name} already exists")
        except Exception as e:
            # If repository not found, create it
            print(f"Creating new dataset repository: {dataset_name}")
            api.create_repo(
                repo_id=dataset_name,
                repo_type="dataset",
                private=private,
                exist_ok=True
            )
        
        # Create README content
        readme_content = create_readme(
            dataset_name=dataset_name,
            dataset_dict=dataset_dict,
            class_names=class_names,
            few_shot_configs=few_shot_configs,
            partition_configs=partition_configs
        )
        
        # Write README to a temporary file
        with open("README.md", "w") as f:
            f.write(readme_content)
        
        # Push dataset to Hub
        print(f"Pushing dataset to Hugging Face Hub: {dataset_name}")
        dataset_dict.push_to_hub(
            dataset_name,
            token=token,
            private=private
        )
        
        # Upload README file
        print("Uploading README.md file")
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=dataset_name,
            repo_type="dataset"
        )
        
        # Upload source data files if available
        if annotation_file and os.path.exists(annotation_file):
            print(f"Uploading annotation file: {annotation_file}")
            api.upload_file(
                path_or_fileobj=annotation_file,
                path_in_repo=os.path.basename(annotation_file),
                repo_id=dataset_name,
                repo_type="dataset"
            )
        
        if mapping_file and os.path.exists(mapping_file):
            print(f"Uploading mapping file: {mapping_file}")
            api.upload_file(
                path_or_fileobj=mapping_file,
                path_in_repo=os.path.basename(mapping_file),
                repo_id=dataset_name,
                repo_type="dataset"
            )
        
        # Clean up temporary files
        if os.path.exists("README.md"):
            os.remove("README.md")
            
        print(f"Successfully uploaded dataset to {dataset_name}")
        
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up any temporary files
        if os.path.exists("README.md"):
            os.remove("README.md") 
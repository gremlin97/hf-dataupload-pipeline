"""
Core functionality for uploading Mars multi-label datasets to Hugging Face.
"""

import os
import json
import ast
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime

import pandas as pd
from datasets import Dataset, DatasetDict, Image, ClassLabel, Features, DatasetInfo, Value, Sequence
from tqdm import tqdm
import huggingface_hub
from huggingface_hub.utils import RepositoryNotFoundError
import glob

# Default dataset information
DEFAULT_DATASET_INFO = {
    "description": "MER - Mars Exploration Rover Multi-Label Classification Dataset",
    "license": "cc-by-4.0",
    "tags": ["mars", "multi-label-classification", "mer", "rover"],
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
    
    # For multi-label, we don't use ClassLabel since we have multiple labels per example
    
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
    feature_name_column: str = "feature_name",
    split_column: str = "split",
    mapping_file: Optional[str] = None,
    few_shot_files: Optional[List[str]] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
) -> Tuple[DatasetDict, List[Dict], List[str], Optional[Dict]]:
    """
    Create a DatasetDict from the Mars multi-label dataset annotation file and image directory.
    
    Args:
        annotation_file: Path to the annotation CSV file
        data_dir: Path to the data directory containing images
        text_column: Name of the image path column
        label_column: Name of the label column
        feature_name_column: Name of the column containing feature names
        split_column: Name of the split column
        mapping_file: Path to the mapping file
        few_shot_files: List of paths to few-shot CSV files (optional)
        dataset_info: Optional dictionary containing dataset metadata
    
    Returns:
        Tuple of (DatasetDict containing the datasets, 
                 List of few-shot configurations,
                 List of class names,
                 Mapping dictionary from class index to name)
    """
    print(f"Attempting to read annotation file: {annotation_file}")
    
    all_few_shot_configs = []
    dataset_dict = {}
    mapping_dict = None
    class_names = []
    
    # First try to load mapping file if available to get class names
    if mapping_file and os.path.exists(mapping_file):
        try:
            print(f"Loading class names from mapping file: {mapping_file}")
            with open(mapping_file, 'r') as f:
                mapping_dict = json.load(f)
                print(f"Loaded mapping file with {len(mapping_dict)} classes")
                
            # Extract class names from mapping, preserving the order
            max_class_id = max(int(idx) for idx in mapping_dict.keys())
            class_names = ["" for _ in range(max_class_id + 1)]
            for idx, name_info in mapping_dict.items():
                # In case the mapping format is different, handle both formats
                if isinstance(name_info, list) and len(name_info) > 0:
                    class_names[int(idx)] = name_info[0]  # Short name
                else:
                    class_names[int(idx)] = name_info
            
            print(f"Extracted {len(class_names)} class names from mapping file")
        except Exception as e:
            print(f"Error loading mapping file: {e}")
            import traceback
            traceback.print_exc()
    
    # Check if annotation file exists
    if not os.path.exists(annotation_file):
        raise ValueError(f"Annotation file '{annotation_file}' not found.")
    
    # Read annotation file
    try:
        df = pd.read_csv(annotation_file)
        print(f"Loaded annotation file with {len(df)} rows and {df.columns.tolist()} columns")
        
        # For multi-label classification, we need to process the label column differently
        # Each row has a list of labels in string format that we need to convert to actual lists
        
        # Count the total number of unique labels in the dataset if not already from mapping
        if not class_names:
            all_labels = set()
            for labels_str in df[label_column]:
                try:
                    # Handle different formats of label representation
                    if isinstance(labels_str, str):
                        # Try parsing as literal Python list
                        labels = ast.literal_eval(labels_str)
                        all_labels.update(labels)
                except (ValueError, SyntaxError):
                    print(f"Warning: Could not parse labels: {labels_str}")
            
            class_names = [f"class_{i}" for i in range(max(all_labels) + 1)]
            print(f"Inferred {len(class_names)} class names from label data")
        
        # Process labels to ensure they are lists of integers
        df['labels_processed'] = df[label_column].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        # Process feature names to ensure they are lists of strings
        if feature_name_column in df.columns:
            df['features_processed'] = df[feature_name_column].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        
        # Create standard splits
        for split in ['train', 'val', 'test']:
            split_df = df[df[split_column] == split].copy()
            if not split_df.empty:
                print(f"Creating {split} split with {len(split_df)} examples")
                
                # Create multi-hot encoded label vectors
                multi_hot_labels = []
                for _, row in split_df.iterrows():
                    labels = row['labels_processed']
                    # Create a multi-hot vector (binary vector where 1 indicates class presence)
                    multi_hot = [1 if i in labels else 0 for i in range(len(class_names))]
                    multi_hot_labels.append(multi_hot)
                
                # Create the image column with full paths
                image_paths = [os.path.join(data_dir, split, x) for x in split_df[text_column]]
                
                # Verify images exist
                missing_images = 0
                valid_image_paths = []
                for img_path in image_paths:
                    if not os.path.exists(img_path):
                        missing_images += 1
                    else:
                        valid_image_paths.append(img_path)
                
                if missing_images > 0:
                    print(f"Warning: {missing_images} images not found in {split} split")
                
                data_dict = {
                    'image': valid_image_paths,
                    'labels': multi_hot_labels
                }
                
                # Add feature names if available
                if feature_name_column in df.columns:
                    data_dict['feature_names'] = split_df['features_processed'].tolist()
                
                # Define features with Image type for actual image loading
                features = Features({
                    'image': Image(),
                    'labels': Sequence(Value('int8')),
                })
                
                if feature_name_column in df.columns:
                    features['feature_names'] = Sequence(Value('string'))
                
                # Create dataset with image features to load actual images
                dataset_dict[split] = Dataset.from_dict(data_dict, features=features)
                print(f"Created {split} split with {len(dataset_dict[split])} examples and loaded images")
    
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Could not create dataset from annotation file: {str(e)}")
    
    # Add few-shot splits if provided
    if few_shot_files:
        for few_shot_file in few_shot_files:
            if os.path.exists(few_shot_file):
                try:
                    print(f"Processing few-shot file: {few_shot_file}")
                    few_shot_df = pd.read_csv(few_shot_file)
                    
                    # Store file info for config
                    file_basename = os.path.basename(few_shot_file)
                    few_shot_config = {"filename": file_basename}
                    all_few_shot_configs.append(few_shot_config)
                    
                    # Create few-shot split name based on file name
                    file_prefix = os.path.splitext(file_basename)[0]
                    few_shot_split_name = f"few_shot_train_{file_prefix}"
                    
                    # Process labels to ensure they are lists of integers
                    few_shot_df['labels_processed'] = few_shot_df[label_column].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                    )
                    
                    # Create the image paths with full paths
                    few_shot_df['image_path'] = few_shot_df[text_column].apply(
                        lambda x: os.path.join(data_dir, 'train', x)
                    )
                    
                    # Filter to only include existing images
                    valid_df = few_shot_df[few_shot_df['image_path'].apply(os.path.exists)].copy()
                    missing_count = len(few_shot_df) - len(valid_df)
                    if missing_count > 0:
                        print(f"Warning: {missing_count} images not found in few-shot split")
                    
                    if valid_df.empty:
                        print(f"Error: No valid images found in few-shot file {few_shot_file}")
                        continue
                    
                    # Create multi-hot encoded label vectors (only for valid images)
                    multi_hot_labels = []
                    for _, row in valid_df.iterrows():
                        labels = row['labels_processed']
                        # Create a multi-hot vector (binary vector where 1 indicates class presence)
                        multi_hot = [1 if i in labels else 0 for i in range(len(class_names))]
                        multi_hot_labels.append(multi_hot)
                    
                    # Create data dictionary with only valid images and matching labels
                    data_dict = {
                        'image': valid_df['image_path'].tolist(),
                        'labels': multi_hot_labels
                    }
                    
                    # Add feature names if available and in the few-shot file
                    if feature_name_column in few_shot_df.columns:
                        valid_df['features_processed'] = valid_df[feature_name_column].apply(
                            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                        )
                        data_dict['feature_names'] = valid_df['features_processed'].tolist()
                    
                    # Define features with Image type for actual image loading
                    features = Features({
                        'image': Image(),
                        'labels': Sequence(Value('int8')),
                    })
                    
                    if feature_name_column in few_shot_df.columns:
                        features['feature_names'] = Sequence(Value('string'))
                    
                    # Debug print to confirm lengths
                    print(f"Few-shot data: {len(data_dict['image'])} images, {len(data_dict['labels'])} label sets")
                    
                    # Create dataset with image features to load actual images
                    dataset_dict[few_shot_split_name] = Dataset.from_dict(data_dict, features=features)
                    print(f"Created {few_shot_split_name} split with {len(dataset_dict[few_shot_split_name])} examples and loaded images")
                
                except Exception as e:
                    print(f"Error processing few-shot file {few_shot_file}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: Few-shot file not found: {few_shot_file}")
    
    if not dataset_dict:
        raise ValueError("No valid data found in any split. Please check your dataset structure and annotation file.")
    
    return DatasetDict(dataset_dict), all_few_shot_configs, class_names, mapping_dict


def create_readme(
    dataset_name: str,
    dataset_dict: DatasetDict,
    class_names: List[str],
    mapping_dict: Optional[Dict] = None,
    few_shot_configs: Optional[List[Dict]] = None
) -> str:
    """
    Create README content for the dataset with YAML frontmatter.
    
    Args:
        dataset_name: Name of the dataset
        dataset_dict: DatasetDict to extract statistics from
        class_names: List of class names
        mapping_dict: Dictionary mapping class indices to names
        few_shot_configs: List of few-shot configuration dictionaries

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
    readme += "size_categories:\n- 1K<n<10K\n"
    readme += "source_datasets:\n- original\n"
    readme += "task_categories:\n- image-classification\n"
    readme += "task_ids:\n- multi-label-image-classification\n"
    readme += f"pretty_name: MER - Mars Exploration Rover Dataset\n"
    
    # End frontmatter
    readme += "---\n\n"
    
    # Add dataset title
    readme += f"# MER - Mars Exploration Rover Dataset\n\n"
    
    # Add dataset description
    readme += "A multi-label classification dataset containing Mars images from the Mars Exploration Rover (MER) mission for planetary science research.\n\n"
    
    # Add metadata section
    readme += "## Dataset Metadata\n\n"
    
    # Add license, version, datePublished, and citeAs
    readme += "* **License:** CC-BY-4.0 (Creative Commons Attribution 4.0 International)\n"
    readme += "* **Version:** 1.0\n"
    readme += f"* **Date Published:** {current_date}\n"
    readme += "* **Cite As:** TBD\n\n"
    
    # Add class information
    readme += "## Classes\n\n"
    readme += "This dataset uses multi-label classification, meaning each image can have multiple class labels.\n\n"
    
    if mapping_dict:
        readme += "The dataset contains the following classes:\n\n"
        for class_idx, class_info in mapping_dict.items():
            if isinstance(class_info, list) and len(class_info) >= 2:
                # Format: [short_name, description]
                readme += f"- **{class_info[0]}** ({class_idx}): {class_info[1]}\n"
            elif isinstance(class_info, list) and len(class_info) == 1:
                # Format: [short_name]
                readme += f"- **{class_info[0]}** ({class_idx})\n"
            else:
                # Just use class_info directly
                readme += f"- **{class_info}** ({class_idx})\n"
    elif class_names:
        readme += "The dataset contains the following classes:\n\n"
        class_list = "\n".join([f"- **{name}** ({i})" for i, name in enumerate(class_names) if name])
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
    
    # Add format information
    readme += "## Format\n\n"
    readme += "Each example in the dataset has the following format:\n\n"
    readme += "```\n"
    readme += "{\n"
    readme += "  'image': Image(...),  # PIL image\n"
    readme += "  'labels': List[int],  # Multi-hot encoded binary vector (1 if class is present, 0 otherwise)\n"
    
    if 'feature_names' in next(iter(dataset_dict.values())).features:
        readme += "  'feature_names': List[str],  # List of feature names (class short codes)\n"
    
    readme += "}\n"
    readme += "```\n\n"
    
    # Add usage example
    readme += "## Usage\n\n"
    readme += "```python\n"
    readme += "from datasets import load_dataset\n\n"
    readme += f"dataset = load_dataset(\"{dataset_name}\")\n\n"
    readme += "# Access an example\n"
    readme += "example = dataset['train'][0]\n"
    readme += "image = example['image']  # PIL image\n"
    readme += "labels = example['labels']  # Multi-hot encoded binary vector\n\n"
    
    readme += "# Example of how to find which classes are present in an image\n"
    readme += "present_classes = [i for i, is_present in enumerate(labels) if is_present == 1]\n"
    readme += "print(f\"Classes present in this image: {present_classes}\")\n"
    readme += "```\n\n"
    
    # Add information about multi-label classification
    readme += "## Multi-label Classification\n\n"
    readme += "In multi-label classification, each image can belong to multiple classes simultaneously. "
    readme += "The labels are represented as a binary vector where a 1 indicates the presence of a class "
    readme += "and a 0 indicates its absence.\n\n"
    
    readme += "Unlike single-label classification where each image has exactly one class, "
    readme += "multi-label classification allows modeling scenarios where multiple features can "
    readme += "be present in the same image, which is often the case with Mars imagery.\n"
    
    return readme


def upload_to_huggingface(
    dataset_dict: DatasetDict,
    dataset_name: str,
    token: str,
    private: bool = False,
    dataset_info: Optional[Dict[str, Any]] = None,
    class_names: Optional[List[str]] = None,
    mapping_dict: Optional[Dict] = None,
    few_shot_configs: Optional[List[Dict]] = None,
    annotation_file: Optional[str] = None,
    mapping_file: Optional[str] = None
) -> None:
    """
    Upload a multi-label dataset to Hugging Face.
    
    Args:
        dataset_dict: DatasetDict to upload
        dataset_name: Name for the dataset on Hugging Face
        token: Hugging Face API token
        private: Whether to make the dataset private
        dataset_info: Optional dictionary containing dataset metadata
        class_names: List of class names for the dataset
        mapping_dict: Dictionary mapping class indices to names
        few_shot_configs: List of few-shot configuration dictionaries
        annotation_file: Path to the annotation file to upload (optional)
        mapping_file: Path to the mapping file to upload (optional)
    """
    info = dataset_info or DEFAULT_DATASET_INFO.copy()
    
    # Prepare commit description
    description = info.get("description", "MER - Mars Exploration Rover Multi-Label Classification Dataset")
    if class_names:
        description += f"\nClasses: {len(class_names)} classes"
        
    # Add info about few-shot splits
    few_shot_splits = [split for split in dataset_dict.keys() if 'few_shot_train_' in split]
    if few_shot_splits:
        description += f"\n\n--- Few-shot Splits ---"
        description += f"\nDetected splits: {', '.join(few_shot_splits)}"
        if few_shot_configs:
             description += "\n\nConfigurations:" 
             for fs_config_info in few_shot_configs:
                 filename = fs_config_info.get("filename", "unknown_file")
                 description += f"\n\n* File: {filename}"

    # Initialize the API client
    api = huggingface_hub.HfApi()

    # Check if repository exists, create it if not
    try:
        api.repo_info(repo_id=dataset_name, repo_type="dataset")
    except Exception as e:
        # Handle multiple potential exception types
        print(f"Repository {dataset_name} not found. Creating new repository...")
        try:
            api.create_repo(repo_id=dataset_name, repo_type="dataset", private=private, token=token)
        except Exception as create_error:
            print(f"Error creating repository: {create_error}")
            raise

    # Upload to hub
    dataset_dict.push_to_hub(
        repo_id=dataset_name,
        token=token,
        private=private,
        commit_message="Upload MER - Mars Exploration Rover multi-label classification dataset",
        commit_description=description
    )
    
    # Create and upload README.md
    readme_content = create_readme(
        dataset_name=dataset_name,
        dataset_dict=dataset_dict,
        class_names=class_names,
        mapping_dict=mapping_dict,
        few_shot_configs=few_shot_configs
    )
    readme_path = "README.md.tmp"
    with open(readme_path, "w") as f:
        f.write(readme_content)
        
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=dataset_name,
        repo_type="dataset",
        token=token,
        commit_message="Add dataset README with metadata"
    )
    
    # Upload annotation and mapping files if provided
    if annotation_file or mapping_file:
        try:
            if annotation_file and os.path.exists(annotation_file):
                annotation_filename = os.path.basename(annotation_file)
                print(f"Uploading annotation file: {annotation_filename}")
                api.upload_file(
                    path_or_fileobj=annotation_file,
                    path_in_repo=annotation_filename,  # Upload to root directory
                    repo_id=dataset_name,
                    repo_type="dataset",
                    token=token,
                    commit_message=f"Add annotation file: {annotation_filename}"
                )
            
            if mapping_file and os.path.exists(mapping_file):
                mapping_filename = os.path.basename(mapping_file)
                print(f"Uploading mapping file: {mapping_filename}")
                api.upload_file(
                    path_or_fileobj=mapping_file,
                    path_in_repo=mapping_filename,  # Upload to root directory
                    repo_id=dataset_name,
                    repo_type="dataset",
                    token=token,
                    commit_message=f"Add mapping file: {mapping_filename}"
                )
        except Exception as e:
            print(f"Error uploading additional files: {str(e)}")
            print("The dataset was uploaded successfully, but there was an error uploading the additional files.")
    
    # Clean up temporary files
    if os.path.exists(readme_path):
        os.remove(readme_path)
    
    print(f"Successfully uploaded dataset to: {dataset_name}") 
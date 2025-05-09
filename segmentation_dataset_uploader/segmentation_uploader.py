"""
Segmentation Dataset Uploader - A utility for uploading segmentation datasets to Hugging Face.
Supports image segmentation with masks.
"""

import os
import json
import glob
import shutil
import yaml
import argparse
from typing import Dict, List, Optional, Any, Tuple

try:
    import numpy as np
    import pandas as pd
    from datasets import Dataset, DatasetDict, Image, Features, Value, Sequence
    from huggingface_hub import HfApi, HfFolder
    from PIL import Image as PILImage
except ImportError:
    raise ImportError("Required packages not found. Please install: datasets, huggingface_hub, pandas, numpy, pillow, pyyaml")

class SegmentationUploader:
    """Class to handle uploading segmentation datasets to Hugging Face."""
    
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        token: Optional[str] = None,
        config_file: Optional[str] = None,
        mapping_file: Optional[str] = None,
        private: bool = False,
    ):
        """
        Initialize the uploader.
        
        Args:
            data_dir: Path to the data directory containing images and masks
            dataset_name: Name for the dataset on Hugging Face (username/dataset-name)
            token: Hugging Face API token (will use stored token if None)
            config_file: Path to configuration file (optional)
            mapping_file: Path to mapping.json file (optional)
            private: Whether to make the dataset private
        """
        self.data_dir = os.path.abspath(data_dir)
        print(f"Using data directory: {self.data_dir}")
        
        self.dataset_name = dataset_name
        self.token = token or HfFolder.get_token()
        self.config_file = config_file
        
        # Convert to absolute path if provided
        if mapping_file:
            self.mapping_file = os.path.abspath(mapping_file)
            print(f"Using mapping file: {self.mapping_file}")
        else:
            self.mapping_file = None
            
        self.private = private
        self.class_names = None
        self.config = {}
        
        # If config file provided, load it
        if self.config_file:
            self._load_config_from_yaml()
        # Otherwise, try to find mapping.json or infer class names
        else:
            self._infer_config_from_directory()
        
        # Initialize API
        self.api = HfApi(token=self.token)

    def _load_config_from_yaml(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
                
            # Get class names from config
            self.class_names = self.config.get('names', [])
            if not self.class_names:
                print("Warning: No class names found in config file. Will try to infer from mapping file.")
                self._load_class_names_from_mapping()
        except Exception as e:
            print(f"Error loading config file: {e}")
            self._infer_config_from_directory()

    def _infer_config_from_directory(self):
        """Infer configuration from directory structure and mapping file."""
        # Set path to the parent directory of data_dir
        parent_dir = os.path.dirname(self.data_dir)
        self.config['path'] = os.path.basename(parent_dir)
        
        # Try to find mapping.json if not provided
        if not self.mapping_file:
            potential_mapping = os.path.join(parent_dir, "mapping.json")
            if os.path.exists(potential_mapping):
                self.mapping_file = potential_mapping
                print(f"Found mapping file: {self.mapping_file}")
            else:
                print(f"No mapping file found at {potential_mapping}")
        
        # Load class names from mapping file
        self._load_class_names_from_mapping()
        
        # Auto-detect split directories
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.data_dir, split)
            print(f"Checking for {split} directory at: {split_dir}")
            
            if os.path.isdir(split_dir):
                images_dir = os.path.join(split_dir, 'images')
                print(f"Checking for images directory at: {images_dir}")
                
                if os.path.isdir(images_dir):
                    self.config[split] = f"{split}/images"
                    print(f"Detected {split} directory: {images_dir}")

    def _load_class_names_from_mapping(self):
        """Load class names from mapping.json file."""
        if self.mapping_file and os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r') as f:
                    mapping = json.load(f)
                
                print(f"Loaded mapping file content: {mapping}")
                
                # Sort by class index to ensure correct order
                max_class_id = max(int(idx) for idx in mapping.keys())
                self.class_names = ["" for _ in range(max_class_id + 1)]
                
                for idx, name in mapping.items():
                    self.class_names[int(idx)] = name
                
                # Set number of classes
                self.config['nc'] = len(self.class_names)
                self.config['names'] = self.class_names
                
                print(f"Loaded {len(self.class_names)} classes from mapping file: {self.class_names}")
            except Exception as e:
                print(f"Error loading mapping file: {e}")
                self._fallback_class_names()
        else:
            if self.mapping_file:
                print(f"Mapping file not found at: {self.mapping_file}")
            else:
                print("No mapping file provided")
            self._fallback_class_names()

    def _fallback_class_names(self):
        """Create fallback class names if no mapping file is found."""
        print("Warning: No mapping file found. Using generic class names.")
        self.class_names = ['Background', 'Object']
        self.config['nc'] = len(self.class_names)
        self.config['names'] = self.class_names

    def create_segmentation_dataset(self) -> DatasetDict:
        """
        Create a dataset dictionary from segmentation images and masks.
        
        Returns:
            DatasetDict containing the datasets
        """
        dataset_dict = {}
        
        if not self.class_names:
            raise ValueError("No class names were found or provided. Please provide a config file or mapping file.")
        
        # Define features for segmentation dataset
        features = Features({
            'image': Image(),
            'mask': Image(),
            'width': Value('int64'),
            'height': Value('int64'),
            'class_labels': Sequence(Value('string')),
        })
        
        # Process each split (train, val, test)
        for split in ['train', 'val', 'test']:
            print(f"\nProcessing {split} split...")
            
            # Direct path to images directory for this split
            images_dir = os.path.join(self.data_dir, split, 'images')
            masks_dir = os.path.join(self.data_dir, split, 'masks')
            
            print(f"Checking for images at: {images_dir}")
            print(f"Checking for masks at: {masks_dir}")
            
            if not os.path.isdir(images_dir):
                print(f"Warning: Images directory {images_dir} not found. Skipping {split} split.")
                continue
                
            if not os.path.isdir(masks_dir):
                print(f"Warning: Masks directory {masks_dir} not found. Skipping {split} split.")
                continue
            
            print(f"Found valid directories for {split} split.")
            print(f"Processing {split} split. Images: {images_dir}, Masks: {masks_dir}")
            
            # Get all image files - include .tif files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                found_files = glob.glob(os.path.join(images_dir, f'*{ext}'))
                image_files.extend(found_files)
            
            print(f"Found {len(image_files)} image files in {images_dir}")
            
            # Create dataset examples
            examples = []
            for image_path in image_files:
                image_filename = os.path.basename(image_path)
                base_name, ext = os.path.splitext(image_filename)
                
                # Try both .png and original extension for mask files
                mask_path = os.path.join(masks_dir, f"{base_name}.png")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(masks_dir, image_filename)
                
                # Skip if mask file doesn't exist
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file for {image_filename} not found at {mask_path}. Skipping.")
                    continue
                
                # Get image dimensions
                try:
                    with PILImage.open(image_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"Warning: Could not read image dimensions for {image_path}: {e}. Skipping.")
                    continue
                
                # Find unique class labels in the mask
                try:
                    mask = np.array(PILImage.open(mask_path))
                    unique_classes = np.unique(mask)
                    class_labels = [self.class_names[cls_id] if cls_id < len(self.class_names) else f"unknown_{cls_id}" 
                                   for cls_id in unique_classes if cls_id < 255]  # Exclude 255 which is often used as ignore index
                except Exception as e:
                    print(f"Warning: Could not process mask for {mask_path}: {e}. Skipping.")
                    continue
                
                example = {
                    'image': image_path,
                    'mask': mask_path,
                    'width': width,
                    'height': height,
                    'class_labels': class_labels
                }
                
                examples.append(example)
            
            # Create dataset for this split
            if examples:
                dataset_dict[split] = Dataset.from_list(examples, features=features)
                print(f"Created {split} dataset with {len(examples)} examples")
            else:
                print(f"Warning: No valid examples found for split: {split}")
        
        if not dataset_dict:
            raise ValueError("No valid data found in any split (train/val/test). Please check your directory structure.")
            
        return DatasetDict(dataset_dict)

    def prepare_dataset(self) -> DatasetDict:
        """
        Prepare the dataset based on the format.
        
        Returns:
            DatasetDict with the prepared dataset
        """
        return self.create_segmentation_dataset()
        
    def _create_readme(self, dataset_dict: Optional[DatasetDict] = None) -> str:
        """
        Create README content for the dataset.
        
        Args:
            dataset_dict: Optional dataset dictionary to get statistics from
        
        Returns:
            String containing the README content
        """
        # Get current date for metadata
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Determine available splits
        detected_splits = []
        if dataset_dict:
            detected_splits = list(dataset_dict.keys())
        
        # Get the basename of the dataset
        dataset_basename = self.dataset_name.split('/')[-1]
        
        # Create YAML frontmatter for the README with metadata
        readme = "---\n"
        readme += "annotations_creators:\n- expert-generated\n"
        readme += "language_creators:\n- found\n"
        readme += "language:\n- en\n"
        readme += "license:\n- cc-by-4.0\n"
        readme += "multilinguality:\n- monolingual\n"
        readme += "size_categories:\n- 10K<n<100K\n"
        readme += "source_datasets:\n- original\n"
        readme += "task_categories:\n- image-segmentation\n"
        readme += "task_ids:\n- semantic-segmentation\n"
        readme += f"pretty_name: {dataset_basename}\n"
        readme += "---\n\n"
        
        # Add dataset title
        readme += f"# {dataset_basename}\n\n"
        
        # Add dataset description
        readme += "A segmentation dataset for planetary science applications.\n\n"
        
        # Add metadata section
        readme += "## Dataset Metadata\n\n"
        
        # Add license, version, datePublished, and citeAs
        readme += "* **License:** CC-BY-4.0 (Creative Commons Attribution 4.0 International)\n"
        readme += "* **Version:** 1.0\n"
        readme += f"* **Date Published:** {current_date}\n"
        readme += "* **Cite As:** TBD\n\n"
        
        # Add class information
        readme += "## Classes\n\n"
        if self.class_names:
            readme += "This dataset contains the following classes:\n\n"
            class_list = "\n".join([f"- {i}: {name}" for i, name in enumerate(self.class_names)])
            readme += f"{class_list}\n\n"
        
        # Add directory structure
        readme += "## Directory Structure\n\n"
        readme += "The dataset follows this structure:\n\n"
        readme += "```\n"
        readme += "dataset/\n"
        
        # Add detected splits or use default structure
        if dataset_dict:
            for split in ['train', 'val', 'test']:
                if split in dataset_dict:
                    readme += f"  ├── {split}/\n"
                    readme += f"  │   ├── images/  # Image files\n"
                    readme += f"  │   └── masks/   # Segmentation masks\n"
        else:
            # Use default structure if dataset_dict is not available
            for split in ['train', 'val', 'test']:
                readme += f"  ├── {split}/\n"
                readme += f"  │   ├── images/  # Image files\n"
                readme += f"  │   └── masks/   # Segmentation masks\n"
        
        # Finish structure
        readme += "```\n\n"
        
        # Add split statistics
        if dataset_dict:
            readme += "## Statistics\n\n"
            for split in detected_splits:
                count = len(dataset_dict[split])
                readme += f"- {split}: {count} images\n"
            readme += "\n"
        
        # Add usage example
        readme += "## Usage\n\n"
        readme += "```python\n"
        readme += "from datasets import load_dataset\n\n"
        readme += f"dataset = load_dataset(\"{self.dataset_name}\")\n"
        readme += "```\n\n"
        
        # Add information about the format
        readme += "## Format\n\n"
        readme += "Each example in the dataset has the following format:\n\n"
        readme += "```\n"
        readme += "{\n"
        readme += "  'image': Image(...),      # PIL image\n"
        readme += "  'mask': Image(...),       # PIL image of the segmentation mask\n"
        readme += "  'width': int,             # Width of the image\n"
        readme += "  'height': int,            # Height of the image\n"
        readme += "  'class_labels': [str,...] # List of class names present in the mask\n"
        readme += "}\n"
        readme += "```\n"
        
        return readme

    def upload(self, dataset_dict: Optional[DatasetDict] = None) -> None:
        """
        Upload the dataset to Hugging Face Hub.
        
        Args:
            dataset_dict: Optional dataset dictionary to upload
        """
        if dataset_dict is None:
            dataset_dict = self.prepare_dataset()
        
        if not dataset_dict:
            raise ValueError("No dataset to upload")
        
        # Prepare description
        description = f"Segmentation dataset with {len(self.class_names)} classes: {', '.join(self.class_names)}"
        
        # Get current date for metadata
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Push to hub
        print(f"Uploading dataset to {self.dataset_name}...")
        
        # Upload to hub
        dataset_dict.push_to_hub(
            self.dataset_name,
            token=self.token,
            private=self.private,
            commit_message="Upload segmentation dataset",
            commit_description=description
        )
        
        # Upload additional files
        
        # If we have a mapping file, upload it
        if self.mapping_file and os.path.exists(self.mapping_file):
            self.api.upload_file(
                path_or_fileobj=self.mapping_file,
                path_in_repo=os.path.basename(self.mapping_file),
                repo_id=self.dataset_name,
                repo_type="dataset"
            )
        
        # If we have a config file, upload it
        if self.config_file and os.path.exists(self.config_file):
            self.api.upload_file(
                path_or_fileobj=self.config_file,
                path_in_repo=os.path.basename(self.config_file),
                repo_id=self.dataset_name,
                repo_type="dataset"
            )
            
        # Create and upload README.md
        readme_content = self._create_readme(dataset_dict)
        readme_path = "README.md.tmp"
        with open(readme_path, "w") as f:
            f.write(readme_content)
            
        self.api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=self.dataset_name,
            repo_type="dataset"
        )
        
        # Clean up temporary files
        if os.path.exists(readme_path):
            os.remove(readme_path)
        
        print(f"Segmentation dataset uploaded successfully to: https://huggingface.co/datasets/{self.dataset_name}")


def main():
    """Command line interface for the segmentation dataset uploader."""
    parser = argparse.ArgumentParser(description="Upload segmentation datasets to Hugging Face Hub")
    parser.add_argument("--data_dir", required=True, help="Directory containing the data")
    parser.add_argument("--dataset_name", required=True, help="Name for the dataset on HF (username/dataset-name)")
    parser.add_argument("--token", help="Hugging Face API token")
    parser.add_argument("--config_file", help="Path to config file (optional)")
    parser.add_argument("--mapping_file", help="Path to mapping.json file (optional)")
    parser.add_argument("--private", action="store_true", help="Make the dataset private")
    
    args = parser.parse_args()
    
    uploader = SegmentationUploader(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        token=args.token,
        config_file=args.config_file,
        mapping_file=args.mapping_file,
        private=args.private
    )
    
    uploader.upload()


if __name__ == "__main__":
    main() 
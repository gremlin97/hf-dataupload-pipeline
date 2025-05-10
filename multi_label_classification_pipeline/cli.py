#!/usr/bin/env python3
"""
Command-line interface for uploading MER (Mars Exploration Rover) multi-label datasets to Hugging Face.
"""

import os
import sys
import argparse
import glob
from typing import List, Optional

from multi_label_classification_pipeline.uploader import (
    create_dataset_dict,
    upload_to_huggingface
)

def get_few_shot_files(few_shot_dir: str) -> List[str]:
    """
    Get list of few-shot CSV files in the directory.
    
    Args:
        few_shot_dir: Directory containing few-shot files
        
    Returns:
        List of paths to few-shot files
    """
    if not os.path.exists(few_shot_dir):
        return []
    
    return sorted(glob.glob(os.path.join(few_shot_dir, "*.csv")))

def main():
    parser = argparse.ArgumentParser(description="Upload MER (Mars Exploration Rover) multi-label dataset to Hugging Face")
    
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Name for the dataset on Hugging Face (e.g., username/dataset-name)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the data directory containing MER images")
    parser.add_argument("--annotation-file", type=str, required=True,
                        help="Path to the annotation CSV file")
    parser.add_argument("--mapping-file", type=str, default=None,
                        help="Path to the mapping file")
    parser.add_argument("--few-shot-dir", type=str, default=None,
                        help="Directory containing few-shot files")
    parser.add_argument("--token", type=str, required=True,
                        help="Hugging Face API token")
    parser.add_argument("--private", action="store_true",
                        help="Make the dataset private")
    parser.add_argument("--text-column", type=str, default="file_id",
                        help="Name of the column containing image filenames")
    parser.add_argument("--label-column", type=str, default="label",
                        help="Name of the column containing labels")
    parser.add_argument("--feature-name-column", type=str, default="feature_name",
                        help="Name of the column containing feature names")
    parser.add_argument("--split-column", type=str, default="split",
                        help="Name of the column containing split information")
    
    args = parser.parse_args()
    
    # Make paths absolute if they're relative
    data_dir = os.path.abspath(args.data_dir)
    annotation_file = os.path.abspath(args.annotation_file)
    mapping_file = os.path.abspath(args.mapping_file) if args.mapping_file else None
    few_shot_dir = os.path.abspath(args.few_shot_dir) if args.few_shot_dir else None
    
    print(f"Uploading MER (Mars Exploration Rover) multi-label dataset to Hugging Face: {args.dataset_name}")
    print(f"Data directory: {data_dir}")
    print(f"Annotation file: {annotation_file}")
    print(f"Mapping file: {mapping_file}")
    
    # Find few-shot files
    few_shot_files = get_few_shot_files(few_shot_dir) if few_shot_dir else []
    if few_shot_files:
        print(f"Found {len(few_shot_files)} few-shot files in {few_shot_dir}")
        for fs_file in few_shot_files:
            print(f"  - {os.path.basename(fs_file)}")
    
    # Create dataset dictionary
    try:
        dataset_dict, few_shot_configs, class_names, mapping_dict = create_dataset_dict(
            annotation_file=annotation_file,
            data_dir=data_dir,
            text_column=args.text_column,
            label_column=args.label_column,
            feature_name_column=args.feature_name_column,
            split_column=args.split_column,
            mapping_file=mapping_file,
            few_shot_files=few_shot_files
        )
        
        # Display information about the dataset
        print("\nMER Dataset Summary:")
        for split_name, split_dataset in dataset_dict.items():
            print(f"  - {split_name}: {len(split_dataset)} examples")
        
        # Upload to Hugging Face
        print(f"\nUploading MER dataset to Hugging Face as {args.dataset_name}")
        upload_to_huggingface(
            dataset_dict=dataset_dict,
            dataset_name=args.dataset_name,
            token=args.token,
            private=args.private,
            class_names=class_names,
            mapping_dict=mapping_dict,
            few_shot_configs=few_shot_configs,
            annotation_file=annotation_file,
            mapping_file=mapping_file
        )
        
        print("\nMER dataset upload complete!")
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
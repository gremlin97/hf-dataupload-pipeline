"""
Command-line interface for the Mars Dataset Uploader.
"""

import argparse
import os
import sys

from classification_pipeline.uploader import create_dataset_dict, upload_to_huggingface


def main():
    """Main entry point for the Mars Dataset Uploader CLI."""
    parser = argparse.ArgumentParser(description="Upload Mars classification dataset to Hugging Face")
    
    # Required arguments
    parser.add_argument("--data_dir", required=True, help="Path to data directory containing images")
    parser.add_argument("--dataset_name", required=True, help="Name for dataset on Hugging Face")
    parser.add_argument("--token", required=True, help="Hugging Face API token")
    
    # Optional arguments
    parser.add_argument("--annotation_file", help="Path to annotation.csv file (optional if using directory structure)")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    parser.add_argument("--class_names", nargs="+", help="List of class names")
    parser.add_argument("--num_classes", type=int, help="Number of classes if class_names not provided")
    parser.add_argument("--text_column", default="image_path", help="Column name for image paths")
    parser.add_argument("--label_column", default="label", help="Column name for labels")
    parser.add_argument("--split_column", default="split", help="Column name for splits")
    parser.add_argument("--few_shot_dir", help="Directory containing few-shot JSON files")
    parser.add_argument("--partition_dir", help="Directory containing partition JSON files")
    parser.add_argument("--mapping_file", help="Path to mapping file to upload")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with additional logging")
    
    args = parser.parse_args()
    
    # Get current working directory (not the package directory)
    cwd = os.getcwd()
    
    # Make data_dir absolute path from the current directory
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(os.path.join(cwd, args.data_dir))
    print(f"Using data directory: {args.data_dir}")
    
    # Check if the data directory exists
    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Data directory '{args.data_dir}' doesn't exist!")
        sys.exit(1)
    
    # If annotation file is provided, make it absolute path
    if args.annotation_file:
        if not os.path.isabs(args.annotation_file):
            args.annotation_file = os.path.abspath(os.path.join(cwd, args.annotation_file))
        print(f"Using annotation file: {args.annotation_file}")
        
        # Check if annotation file exists
        if not os.path.exists(args.annotation_file):
            print(f"Warning: Annotation file '{args.annotation_file}' doesn't exist!")
    else:
        print("No annotation file provided. Will attempt to infer dataset structure from directory.")
    
    # If mapping file is provided, make it absolute path    
    if args.mapping_file:
        if not os.path.isabs(args.mapping_file):
            args.mapping_file = os.path.abspath(os.path.join(cwd, args.mapping_file))
        print(f"Using mapping file: {args.mapping_file}")
        
        # Check if mapping file exists
        if not os.path.exists(args.mapping_file):
            print(f"Warning: Mapping file '{args.mapping_file}' doesn't exist!")
    
    # Find JSON files in the specified directories
    few_shot_files = []
    if args.few_shot_dir:
        if not os.path.isabs(args.few_shot_dir):
            args.few_shot_dir = os.path.abspath(os.path.join(cwd, args.few_shot_dir))
            
        if os.path.isdir(args.few_shot_dir):
            few_shot_files = [os.path.join(args.few_shot_dir, f) 
                            for f in os.listdir(args.few_shot_dir) if f.endswith('.json')]
            print(f"Found {len(few_shot_files)} few-shot files in {args.few_shot_dir}")
        else:
            print(f"Warning: --few_shot_dir provided ('{args.few_shot_dir}') but it is not a valid directory. Skipping.")

    partition_files = []
    if args.partition_dir:
        if not os.path.isabs(args.partition_dir):
            args.partition_dir = os.path.abspath(os.path.join(cwd, args.partition_dir))
            
        if os.path.isdir(args.partition_dir):
            partition_files = [os.path.join(args.partition_dir, f) 
                            for f in os.listdir(args.partition_dir) if f.endswith('.json')]
            print(f"Found {len(partition_files)} partition files in {args.partition_dir}")
        else:
            print(f"Warning: --partition_dir provided ('{args.partition_dir}') but it is not a valid directory. Skipping.")

    # If annotation file not provided, use a placeholder
    annotation_file = args.annotation_file or os.path.join(args.data_dir, "annotation.csv")
    
    # Check the structure of the data directory
    print("\nAnalyzing data directory structure...")
    found_dirs = []
    for root, dirs, files in os.walk(args.data_dir):
        rel_path = os.path.relpath(root, args.data_dir)
        if rel_path != ".":
            found_dirs.append(rel_path)
            
    if args.debug:
        print(f"Found directories: {found_dirs}")
    
    # Check if we have the expected structure
    has_split_dirs = any(d in found_dirs for d in ['train', 'test', 'val'])
    if not has_split_dirs:
        print("Warning: Could not find standard split directories (train, test, val) directly under data_dir.")
        print("Detected directory structure:")
        for d in found_dirs[:10]:  # Show only first 10 to avoid clutter
            print(f"  - {d}")
        if len(found_dirs) > 10:
            print(f"  - ... and {len(found_dirs) - 10} more")
    
    try:
        # Create dataset dictionary
        dataset_dict, few_shot_configs, partition_configs, annotation_file, mapping_file = create_dataset_dict(
            annotation_file=annotation_file,
            data_dir=args.data_dir,
            text_column=args.text_column,
            label_column=args.label_column,
            split_column=args.split_column,
            class_names=args.class_names,
            num_classes=args.num_classes,
            few_shot_files=few_shot_files, # Pass list of files
            partition_files=partition_files,  # Pass list of files
            mapping_file=args.mapping_file
        )
        
        # Upload to Hugging Face
        upload_to_huggingface(
            dataset_dict=dataset_dict,
            dataset_name=args.dataset_name,
            token=args.token,
            private=args.private,
            class_names=args.class_names,
            few_shot_configs=few_shot_configs,
            partition_configs=partition_configs,
            annotation_file=annotation_file if os.path.exists(annotation_file) else None,
            mapping_file=mapping_file
        )
        
        print(f"Successfully uploaded dataset to {args.dataset_name}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main() 
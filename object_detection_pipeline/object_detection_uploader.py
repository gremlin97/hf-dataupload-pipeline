"""
Object Detection Uploader - A utility for uploading object detection datasets to Hugging Face.
Supports COCO and YOLO format annotations, with optional Pascal VOC format as additional columns.
"""

import os
import json
import glob
import shutil
import yaml
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple

try:
    import numpy as np
    import pandas as pd
    from datasets import Dataset, DatasetDict, Image, Features, Value, Sequence, ClassLabel, Array2D
    from huggingface_hub import HfApi, HfFolder
    from PIL import Image as PILImage
except ImportError:
    raise ImportError("Required packages not found. Please install: datasets, huggingface_hub, pandas, numpy, pillow, pyyaml")

# Define custom feature dictionaries
def yolo_features():
    return {
        'bbox': Sequence(Sequence(Value('float32'))),
        'category': Sequence(Value('string'))
    }

def pascal_voc_features():
    return {
        'filename': Value('string'),
        'size': {
            'width': Value('int64'),
            'height': Value('int64'),
            'depth': Value('int64')
        },
        'objects': Sequence({
            'name': Value('string'),
            'difficult': Value('int64'),
            'bbox': {
                'xmin': Value('int64'),
                'ymin': Value('int64'),
                'xmax': Value('int64'),
                'ymax': Value('int64')
            }
        })
    }

def coco_annotation_features():
    return {
        'image_id': Value('int64'),
        'annotations': Sequence({
            'id': Value('int64'),
            'image_id': Value('int64'),
            'category_id': Value('int64'),
            'bbox': Sequence(Value('float32')),
            'area': Value('float32'),
            # 'iscrowd': Value('int64')
        })
    }

class ObjectDetectionUploader:
    """Class to handle uploading object detection datasets to Hugging Face."""
    
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        token: Optional[str] = None,
        format: str = "coco",  # "coco" or "yolo"
        config_file: Optional[str] = None,
        partition_dir: Optional[str] = None,
        private: bool = False,
        include_pascal_voc: bool = False,
        include_coco: bool = False,
        include_yolo: bool = False,
    ):
        """
        Initialize the uploader.
        
        Args:
            data_dir: Path to the data directory containing images and annotations
            dataset_name: Name for the dataset on Hugging Face (username/dataset-name)
            token: Hugging Face API token (will use stored token if None)
            format: Format of annotations ("coco" or "yolo")
            config_file: Path to configuration file for YOLO format
            private: Whether to make the dataset private
            include_pascal_voc: Whether to include Pascal VOC format annotations as a column
            include_coco: Whether to include COCO format annotations as a column
            include_yolo: Whether to include YOLO format annotations as a column
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.token = token or HfFolder.get_token()
        self.format = format.lower()
        self.config_file = config_file
        self.partition_dir = partition_dir
        self.private = private
        self.include_pascal_voc = include_pascal_voc
        self.include_coco = include_coco
        self.include_yolo = include_yolo
        
        # Validate input
        if self.format not in ["coco", "yolo"]:
            raise ValueError("Format must be either 'coco' or 'yolo'")
        
        if self.format == "yolo" and not self.config_file:
            # Try to find a config file if not provided
            yaml_files = glob.glob(os.path.join(os.path.dirname(data_dir), "*.yaml"))
            if yaml_files:
                self.config_file = yaml_files[0]
                print(f"Using found YAML config: {self.config_file}")
            else:
                raise ValueError("Config file must be provided for YOLO format")
        
        # Initialize API
        self.api = HfApi(token=self.token)

    def create_coco_dataset(self) -> DatasetDict:
        """
        Create a dataset dictionary from COCO format annotations.
        
        Returns:
            DatasetDict containing the datasets
        """
        dataset_dict = {}
        
        # Define base features for COCO object detection dataset
        features = Features({
            'image': Image(),
            'image_id': Value('int64'),
            'width': Value('int64'),
            'height': Value('int64')
        })
        
        # Add additional annotation format features if requested
        if self.include_yolo:
            features['yolo_annotation'] = yolo_features()

        if self.include_coco:
            features['coco_annotation'] = coco_annotation_features()
        
        if self.include_pascal_voc:
            features['pascal_voc_annotation'] = pascal_voc_features()
        
        # Process each split (train, val, test)
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.data_dir, split)
            
            # Skip if the split directory doesn't exist
            if not os.path.isdir(split_dir):
                print(f"Warning: Split directory {split_dir} not found. Skipping.")
                continue
                
            # Check if COCO annotations exist
            coco_file = os.path.join(split_dir, 'coco_annotations.json')
            
            if not os.path.exists(coco_file):
                print(f"Warning: COCO annotations file not found for split: {split}")
                continue
                
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
                
            # Create mappings for easier data processing
            image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
            image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
            category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
            
            # Group annotations by image_id
            annotations_by_image = {}
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(ann)
            
            # Create dataset examples
            examples = []
            images_dir = os.path.join(split_dir, 'images')
            pascal_voc_dir = os.path.join(split_dir, 'pascal_voc')
            pascal_voc_exists = os.path.isdir(pascal_voc_dir)
            
            for image_id, file_name in image_id_to_file.items():
                width, height = image_id_to_size[image_id]
                image_path = os.path.join(images_dir, file_name)
                
                # Skip if image file doesn't exist
                if not os.path.exists(image_path):
                    print(f"Warning: Image file {image_path} not found. Skipping.")
                    continue
                
                example = {
                    'image': image_path,
                    'image_id': image_id,
                    'width': width,
                    'height': height
                }
                
                # Collect bboxes and categories for conversion
                bboxes = []
                categories = []
                
                # Get annotations to use for other formats
                if image_id in annotations_by_image:
                    for ann in annotations_by_image[image_id]:
                        category_name = category_id_to_name.get(ann['category_id'], 'unknown')
                        bboxes.append(ann['bbox'])
                        categories.append(category_name)
                
                # Add COCO annotation if requested
                if self.include_coco:
                    # Create COCO annotation directly from the current data
                    if image_id in annotations_by_image:
                        coco_ann = {
                            'image_id': image_id,
                            'annotations': annotations_by_image.get(image_id, [])
                        }
                        example['coco_annotation'] = coco_ann
                    else:
                        # Empty annotation
                        example['coco_annotation'] = {'image_id': image_id, 'annotations': []}
                
                # Add Pascal VOC annotation if requested
                if self.include_pascal_voc:
                    pascal_voc_file = os.path.join(pascal_voc_dir, os.path.splitext(file_name)[0] + '.xml')
                    if os.path.exists(pascal_voc_file):
                        with open(pascal_voc_file, 'r') as f:
                            xml_content = f.read()
                            # Convert XML to dictionary
                            pascal_dict = self._xml_to_dict(xml_content)
                            example['pascal_voc_annotation'] = pascal_dict
                    elif bboxes:
                        # If Pascal VOC file doesn't exist, create one from COCO annotations
                        xml_content = self._create_pascal_voc_from_coco(file_name, width, height, 
                                                                        bboxes, categories)
                        pascal_dict = self._xml_to_dict(xml_content)
                        example['pascal_voc_annotation'] = pascal_dict
                    else:
                        # Empty annotation
                        example['pascal_voc_annotation'] = {
                            'filename': file_name,
                            'size': {'width': width, 'height': height, 'depth': 3},
                            'objects': []
                        }
                
                # Add YOLO annotation if requested and bboxes exist
                if self.include_yolo:
                    if bboxes:
                        # Create YOLO annotation with bboxes and categories arrays
                        yolo_dict = {
                            "bbox": [],
                            "category": []
                        }
                        
                        for bbox, category in zip(bboxes, categories):
                            # Convert COCO format [x, y, width, height] to YOLO format [x_center, y_center, width, height] normalized
                            x, y, w, h = bbox
                            x_center = (x + w/2) / width
                            y_center = (y + h/2) / height
                            w_norm = w / width
                            h_norm = h / height
                            
                            yolo_dict["bbox"].append([x_center, y_center, w_norm, h_norm])
                            yolo_dict["category"].append(category)
                        
                        example['yolo_annotation'] = yolo_dict
                    else:
                        # Empty annotation
                        example['yolo_annotation'] = {"bbox": [], "category": []}
                
                examples.append(example)
            
            # Create dataset for this split
            if examples:
                dataset_dict[split] = Dataset.from_list(examples, features=features)
                print(f"Created {split} dataset with {len(examples)} examples")
            else:
                print(f"Warning: No valid examples found for split: {split}")
        
        return DatasetDict(dataset_dict)

    def create_yolo_dataset(self) -> DatasetDict:
        """
        Create a dataset dictionary from YOLO format annotations.
        
        Returns:
            DatasetDict containing the datasets
        """
        dataset_dict = {}
        
        # Load config file
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get class names
        class_names = config.get('names', [])
        if not class_names:
            raise ValueError("No class names found in config file")
        
        # Define base features for YOLO object detection dataset
        features = Features({
            'image': Image(),
            'width': Value('int64'),
            'height': Value('int64')
        })
        
        # Add additional annotation format features if requested
        if self.include_yolo:
            features['yolo_annotation'] = yolo_features()

        if self.include_coco:
            features['coco_annotation'] = coco_annotation_features()
        
        if self.include_pascal_voc:
            features['pascal_voc_annotation'] = pascal_voc_features()
        
        # Process each split (train, val, test)
        for split in ['train', 'val', 'test']:
            split_path = config.get(split)
            if not split_path:
                print(f"Warning: Path for {split} not found in config. Skipping.")
                continue
            
            # Resolve split path
            if os.path.isabs(split_path):
                images_dir = split_path
            else:
                # Handle relative paths in config
                base_path = os.path.dirname(self.config_file)
                images_dir = os.path.normpath(os.path.join(base_path, split_path))
            
            labels_dir = images_dir.replace('images', 'labels')
            pascal_voc_dir = images_dir.replace('images', 'pascal_voc')
            coco_file = os.path.join(os.path.dirname(images_dir), 'coco_annotations.json')
            
            # Check if directories exist
            if not os.path.isdir(images_dir):
                print(f"Warning: Images directory {images_dir} not found. Skipping.")
                continue
                
            if not os.path.isdir(labels_dir):
                print(f"Warning: Labels directory {labels_dir} not found. Skipping.")
                continue
            
            # Load COCO data if needed
            coco_data = None
            if self.include_coco and os.path.exists(coco_file):
                with open(coco_file, 'r') as f:
                    coco_data = json.load(f)
                file_name_to_id = {img['file_name']: img['id'] for img in coco_data['images']}
                annotations_by_image_id = {}
                for ann in coco_data['annotations']:
                    image_id = ann['image_id']
                    if image_id not in annotations_by_image_id:
                        annotations_by_image_id[image_id] = []
                    annotations_by_image_id[image_id].append(ann)
            
            # Get all image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                image_files.extend(glob.glob(os.path.join(images_dir, f'*{ext}')))
            
            # Create dataset examples
            examples = []
            for image_path in image_files:
                image_filename = os.path.basename(image_path)
                base_filename = os.path.splitext(image_filename)[0]
                label_filename = base_filename + '.txt'
                label_path = os.path.join(labels_dir, label_filename)
                
                # Skip if label file doesn't exist
                if not os.path.exists(label_path):
                    print(f"Warning: Label file for {image_filename} not found. Skipping.")
                    continue
                
                # Get image dimensions
                try:
                    with PILImage.open(image_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"Warning: Could not read image dimensions for {image_path}: {e}. Skipping.")
                    continue
                
                example = {
                    'image': image_path,
                    'width': width,
                    'height': height
                }
                
                # Collect bounding boxes and categories for conversion
                yolo_boxes = []
                categories = []
                
                # Read label file and collect annotations for other formats
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id < len(class_names):
                                # Normalized YOLO bbox [x_center, y_center, width, height]
                                x_center, y_center, box_width, box_height = map(float, parts[1:5])
                                yolo_boxes.append([x_center, y_center, box_width, box_height])
                                categories.append(class_names[class_id])
                            else:
                                print(f"Warning: Invalid class ID {class_id} in {label_path}. Skipping annotation.")
                
                # Add raw YOLO annotation if requested
                if self.include_yolo:
                    # Parse YOLO annotation into a dictionary with bbox and category arrays
                    yolo_dict = {
                        "bbox": [],
                        "category": []
                    }
                    
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                if class_id < len(class_names):
                                    x_center, y_center, box_width, box_height = map(float, parts[1:5])
                                    yolo_dict["bbox"].append([x_center, y_center, box_width, box_height])
                                    yolo_dict["category"].append(class_names[class_id])
                    
                    example['yolo_annotation'] = yolo_dict
                
                # Add COCO annotation if requested
                if self.include_coco:
                    if coco_data:
                        image_id = file_name_to_id.get(image_filename)
                        if image_id and image_id in annotations_by_image_id:
                            coco_annotations = {
                                'image_id': image_id,
                                'annotations': annotations_by_image_id[image_id]
                            }
                            example['coco_annotation'] = coco_annotations
                        else:
                            # Create from YOLO annotations
                            coco_dict = self._create_coco_from_yolo_dict(image_filename, width, height, 
                                                                   yolo_boxes, categories)
                            example['coco_annotation'] = coco_dict
                    else:
                        # Create from YOLO annotations
                        coco_dict = self._create_coco_from_yolo_dict(image_filename, width, height, 
                                                                   yolo_boxes, categories)
                        example['coco_annotation'] = coco_dict
                
                # Add Pascal VOC annotation if requested
                if self.include_pascal_voc:
                    pascal_voc_file = os.path.join(pascal_voc_dir, base_filename + '.xml')
                    if os.path.exists(pascal_voc_file):
                        with open(pascal_voc_file, 'r') as f:
                            xml_content = f.read()
                            pascal_dict = self._xml_to_dict(xml_content)
                            example['pascal_voc_annotation'] = pascal_dict
                    elif yolo_boxes:
                        # If Pascal VOC file doesn't exist, create one from YOLO annotations
                        absolute_boxes = []
                        for box in yolo_boxes:
                            x_center, y_center, box_width, box_height = box
                            # Convert normalized YOLO to absolute coordinates
                            x_min = int((x_center - box_width/2) * width)
                            y_min = int((y_center - box_height/2) * height)
                            x_max = int((x_center + box_width/2) * width)
                            y_max = int((y_center + box_height/2) * height)
                            absolute_boxes.append([x_min, y_min, x_max, y_max])
                        
                        xml_content = self._create_pascal_voc(image_filename, width, height, absolute_boxes, categories)
                        pascal_dict = self._xml_to_dict(xml_content)
                        example['pascal_voc_annotation'] = pascal_dict
                    else:
                        # Empty annotation
                        example['pascal_voc_annotation'] = {
                            'filename': image_filename,
                            'size': {'width': width, 'height': height, 'depth': 3},
                            'objects': []
                        }
                
                examples.append(example)
            
            # Create dataset for this split
            if examples:
                dataset_dict[split] = Dataset.from_list(examples, features=features)
                print(f"Created {split} dataset with {len(examples)} examples")
            else:
                print(f"Warning: No valid examples found for split: {split}")
        
        return DatasetDict(dataset_dict)

    def _create_pascal_voc_from_coco(self, filename: str, width: int, height: int, 
                                      bboxes: List[List[float]], categories: List[str]) -> str:
        """
        Create Pascal VOC annotation from COCO format bbox.
        
        Args:
            filename: Image filename
            width: Image width
            height: Image height
            bboxes: List of COCO bboxes [x, y, width, height]
            categories: List of category names
            
        Returns:
            Pascal VOC annotation string
        """
        root = ET.Element("annotation")
        
        folder = ET.SubElement(root, "folder")
        folder.text = "JPEGImages"
        
        file_elem = ET.SubElement(root, "filename")
        file_elem.text = filename
        
        path = ET.SubElement(root, "path")
        path.text = f"../JPEGImages/{filename}"
        
        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"
        
        size = ET.SubElement(root, "size")
        width_elem = ET.SubElement(size, "width")
        width_elem.text = str(width)
        height_elem = ET.SubElement(size, "height")
        height_elem.text = str(height)
        depth = ET.SubElement(size, "depth")
        depth.text = "1"
        
        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"
        
        for i, (bbox, category) in enumerate(zip(bboxes, categories)):
            # COCO format is [x, y, width, height]
            x, y, w, h = bbox
            
            obj = ET.SubElement(root, "object")
            
            name = ET.SubElement(obj, "n")
            name.text = category
            
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            
            bndbox = ET.SubElement(obj, "bndbox")
            
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(x))
            
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(y))
            
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(x + w))
            
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(y + h))
        
        # Convert to string
        return ET.tostring(root, encoding='unicode')

    def _create_pascal_voc(self, filename: str, width: int, height: int, 
                           bboxes: List[List[int]], categories: List[str]) -> str:
        """
        Create Pascal VOC annotation string.
        
        Args:
            filename: Image filename
            width: Image width
            height: Image height
            bboxes: List of absolute bboxes [xmin, ymin, xmax, ymax]
            categories: List of category names
            
        Returns:
            Pascal VOC annotation string
        """
        root = ET.Element("annotation")
        
        folder = ET.SubElement(root, "folder")
        folder.text = "JPEGImages"
        
        file_elem = ET.SubElement(root, "filename")
        file_elem.text = filename
        
        path = ET.SubElement(root, "path")
        path.text = f"../JPEGImages/{filename}"
        
        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"
        
        size = ET.SubElement(root, "size")
        width_elem = ET.SubElement(size, "width")
        width_elem.text = str(width)
        height_elem = ET.SubElement(size, "height")
        height_elem.text = str(height)
        depth = ET.SubElement(size, "depth")
        depth.text = "1"
        
        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"
        
        for i, (bbox, category) in enumerate(zip(bboxes, categories)):
            # bbox format is [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = bbox
            
            obj = ET.SubElement(root, "object")
            
            name = ET.SubElement(obj, "n")
            name.text = category
            
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            
            bndbox = ET.SubElement(obj, "bndbox")
            
            xmin_elem = ET.SubElement(bndbox, "xmin")
            xmin_elem.text = str(xmin)
            
            ymin_elem = ET.SubElement(bndbox, "ymin")
            ymin_elem.text = str(ymin)
            
            xmax_elem = ET.SubElement(bndbox, "xmax")
            xmax_elem.text = str(xmax)
            
            ymax_elem = ET.SubElement(bndbox, "ymax")
            ymax_elem.text = str(ymax)
        
        # Convert to string
        return ET.tostring(root, encoding='unicode')

    def _create_coco_from_yolo(self, filename: str, width: int, height: int, 
                               bboxes: List[List[float]], categories: List[str]) -> str:
        """
        Create COCO annotation from YOLO format bbox.
        
        Args:
            filename: Image filename
            width: Image width
            height: Image height
            bboxes: List of normalized YOLO bboxes [x_center, y_center, width, height]
            categories: List of category names
            
        Returns:
            COCO annotation JSON string
        """
        # Create a fake image ID
        image_id = hash(filename) % 10000
        
        # Create a COCO annotation
        coco_annotations = []
        
        for i, (bbox, category) in enumerate(zip(bboxes, categories)):
            # Convert normalized YOLO to COCO format
            x_center, y_center, box_width, box_height = bbox
            x = (x_center - box_width/2) * width
            y = (y_center - box_height/2) * height
            w = box_width * width
            h = box_height * height
            
            coco_annotations.append({
                'id': i + 1,
                'image_id': image_id,
                'category_id': 1,  # Assuming single category
                'bbox': [x, y, w, h],
                'area': w * h,
                # 'iscrowd': 0
            })
        
        result = {
            'image_id': image_id,
            'annotations': coco_annotations
        }
        
        return json.dumps(result)

    def _create_yolo_from_coco(self, filename: str, width: int, height: int, 
                               bboxes: List[List[float]], categories: List[str]) -> str:
        """
        Create YOLO annotation from COCO format bbox.
        
        Args:
            filename: Image filename
            width: Image width
            height: Image height
            bboxes: List of COCO bboxes [x, y, width, height]
            categories: List of category names
            
        Returns:
            YOLO annotation string
        """
        # Create a mapping of category names to IDs (for YOLO format)
        unique_categories = list(set(categories))
        category_to_id = {name: i for i, name in enumerate(unique_categories)}
        
        # Create YOLO annotations
        yolo_lines = []
        
        for bbox, category in zip(bboxes, categories):
            # Convert COCO format [x, y, width, height] to YOLO format [x_center, y_center, width, height] normalized
            x, y, w, h = bbox
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            w_norm = w / width
            h_norm = h / height
            
            class_id = category_to_id[category]
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        return "\n".join(yolo_lines)

    def _xml_to_dict(self, xml_string: str) -> dict:
        """
        Convert Pascal VOC XML string to a dictionary representation
        
        Args:
            xml_string: XML string in Pascal VOC format
            
        Returns:
            Dictionary representation of the XML
        """
        root = ET.fromstring(xml_string)
        
        result = {
            'filename': root.findtext('filename', ''),
            'size': {
                'width': int(root.findtext('size/width', 0)),
                'height': int(root.findtext('size/height', 0)),
                'depth': int(root.findtext('size/depth', 0))
            },
            'objects': []
        }
        
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            if bbox is not None:
                result['objects'].append({
                    'name': obj.findtext('name', '') or obj.findtext('n', ''),
                    'difficult': int(obj.findtext('difficult', 0)),
                    'bbox': {
                        'xmin': int(bbox.findtext('xmin', 0)),
                        'ymin': int(bbox.findtext('ymin', 0)),
                        'xmax': int(bbox.findtext('xmax', 0)),
                        'ymax': int(bbox.findtext('ymax', 0))
                    }
                })
        
        return result

    def _create_coco_from_yolo_dict(self, filename: str, width: int, height: int, 
                               bboxes: List[List[float]], categories: List[str]) -> dict:
        """
        Create COCO annotation dictionary from YOLO format bbox.
        
        Args:
            filename: Image filename
            width: Image width
            height: Image height
            bboxes: List of normalized YOLO bboxes [x_center, y_center, width, height]
            categories: List of category names
            
        Returns:
            COCO annotation dictionary
        """
        # Create a fake image ID
        image_id = hash(filename) % 10000
        
        # Create a COCO annotation
        coco_annotations = []
        
        for i, (bbox, category) in enumerate(zip(bboxes, categories)):
            # Convert normalized YOLO to COCO format
            x_center, y_center, box_width, box_height = bbox
            x = (x_center - box_width/2) * width
            y = (y_center - box_height/2) * height
            w = box_width * width
            h = box_height * height
            
            coco_annotations.append({
                'id': i + 1,
                'image_id': image_id,
                'category_id': 1,  # Assuming single category
                'bbox': [x, y, w, h],
                'area': w * h,
                # 'iscrowd': 0
            })
        
        return {
            'image_id': image_id,
            'annotations': coco_annotations
        }

    def prepare_dataset(self) -> DatasetDict:
        """
        Prepare the dataset based on the format.
        
        Returns:
            DatasetDict containing the datasets
        """
        if self.format == "coco":
            return self.create_coco_dataset()
        elif self.format == "yolo":
            return self.create_yolo_dataset()
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def create_partition_datasets(self) -> Dict[str, DatasetDict]:
        """
        Create datasets for each partition based on CSV files in the partition directory.
        This is a separate function that creates datasets directly from the data directory,
        without relying on the main dataset.
        
        Returns:
            Dictionary of partition name to DatasetDict mappings
        """
        if not self.partition_dir or not os.path.isdir(self.partition_dir):
            print(f"Warning: Partition directory {self.partition_dir} not found or not specified. Skipping partitions.")
            return {}
            
        # Find all CSV files in the partition directory
        csv_files = glob.glob(os.path.join(self.partition_dir, "*.csv"))
        if not csv_files:
            print(f"Warning: No CSV files found in partition directory {self.partition_dir}. Skipping partitions.")
            return {}
            
        partition_datasets = {}
        
        # Process each CSV file as a partition
        for csv_file in csv_files:
            partition_name = os.path.splitext(os.path.basename(csv_file))[0]
            print(f"Processing partition: {partition_name}")
            
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Check if the CSV has the required columns
                if 'file_id' not in df.columns or 'split' not in df.columns:
                    print(f"Warning: CSV file {csv_file} does not have required columns 'file_id' and 'split'. Skipping.")
                    continue
                    
                # Create a new DatasetDict for this partition
                partition_dict = {}
                
                # Group by split
                for split, group in df.groupby('split'):
                    # Get file IDs for this split and partition
                    file_ids = set(group['file_id'].tolist())
                    
                    # Determine the images directory for this split
                    if self.format == "coco":
                        images_dir = os.path.join(self.data_dir, split, 'images')
                    elif self.format == "yolo":
                        # For YOLO format, get the path from the config file
                        with open(self.config_file, 'r') as f:
                            config = yaml.safe_load(f)
                        
                        split_path = config.get(split)
                        if not split_path:
                            print(f"Warning: Path for {split} not found in config. Skipping.")
                            continue
                        
                        # Resolve split path
                        if os.path.isabs(split_path):
                            images_dir = split_path
                        else:
                            # Handle relative paths in config
                            base_path = os.path.dirname(self.config_file)
                            images_dir = os.path.normpath(os.path.join(base_path, split_path))
                    else:
                        print(f"Warning: Unsupported format {self.format}. Skipping.")
                        continue
                    
                    # Check if the images directory exists
                    if not os.path.isdir(images_dir):
                        print(f"Warning: Images directory {images_dir} not found. Skipping.")
                        continue
                    
                    # Get the labels directory
                    labels_dir = images_dir.replace('images', 'labels')
                    
                    # Collect examples for this partition
                    examples = []
                    
                    # For each file ID in this partition
                    for file_id in file_ids:
                        # Find the image file
                        image_path = os.path.join(images_dir, file_id)
                        if not os.path.exists(image_path):
                            print(f"Warning: Image file {image_path} not found. Skipping.")
                            continue
                        
                        # Get image dimensions
                        try:
                            with PILImage.open(image_path) as img:
                                width, height = img.size
                        except Exception as e:
                            print(f"Warning: Could not read image dimensions for {image_path}: {e}. Skipping.")
                            continue
                        
                        # Create a basic example
                        example = {
                            'image': image_path,
                            'width': width,
                            'height': height
                        }
                        
                        # Add annotations based on format
                        if self.format == "yolo":
                            # Get the label file
                            base_filename = os.path.splitext(file_id)[0]
                            label_path = os.path.join(labels_dir, base_filename + '.txt')
                            
                            if os.path.exists(label_path):
                                # Read class names from config
                                with open(self.config_file, 'r') as f:
                                    config = yaml.safe_load(f)
                                class_names = config.get('names', [])
                                
                                # Collect bounding boxes and categories for conversion
                                yolo_boxes = []
                                categories = []
                                
                                with open(label_path, 'r') as f:
                                    for line in f:
                                        parts = line.strip().split()
                                        if len(parts) >= 5:
                                            class_id = int(parts[0])
                                            if class_id < len(class_names):
                                                x_center, y_center, box_width, box_height = map(float, parts[1:5])
                                                yolo_boxes.append([x_center, y_center, box_width, box_height])
                                                categories.append(class_names[class_id])
                                
                                # Add YOLO annotations if requested
                                if self.include_yolo:
                                    yolo_dict = {"bbox": [], "category": []}
                                    for box, category in zip(yolo_boxes, categories):
                                        yolo_dict["bbox"].append(box)
                                        yolo_dict["category"].append(category)
                                    example['yolo_annotation'] = yolo_dict
                                
                                # Add COCO annotations if requested
                                if self.include_coco:
                                    # Create a unique image ID (can be the hash of the filename)
                                    import hashlib
                                    image_id = int(hashlib.md5(file_id.encode()).hexdigest(), 16) % (10 ** 10)
                                    
                                    # Convert YOLO boxes to COCO format
                                    coco_annotations = []
                                    for i, (box, category) in enumerate(zip(yolo_boxes, categories)):
                                        x_center, y_center, box_width, box_height = box
                                        
                                        # Convert normalized YOLO to COCO format [x, y, width, height]
                                        x = (x_center - box_width/2) * width
                                        y = (y_center - box_height/2) * height
                                        w = box_width * width
                                        h = box_height * height
                                        
                                        # Get category ID (using index in class_names)
                                        category_id = class_names.index(category)
                                        
                                        # Calculate area
                                        area = w * h
                                        
                                        coco_annotations.append({
                                            'id': i + 1,  # Annotation ID
                                            'image_id': image_id,
                                            'category_id': category_id,
                                            'bbox': [float(x), float(y), float(w), float(h)],
                                            'area': float(area)
                                        })
                                    
                                    example['coco_annotation'] = {
                                        'image_id': image_id,
                                        'annotations': coco_annotations
                                    }
                                
                                # Add Pascal VOC annotations if requested
                                if self.include_pascal_voc:
                                    # Convert YOLO boxes to Pascal VOC format
                                    pascal_objects = []
                                    for box, category in zip(yolo_boxes, categories):
                                        x_center, y_center, box_width, box_height = box
                                        
                                        # Convert normalized YOLO to absolute coordinates
                                        x_min = int((x_center - box_width/2) * width)
                                        y_min = int((y_center - box_height/2) * height)
                                        x_max = int((x_center + box_width/2) * width)
                                        y_max = int((y_center + box_height/2) * height)
                                        
                                        pascal_objects.append({
                                            'name': category,
                                            'difficult': 0,
                                            'bbox': {
                                                'xmin': x_min,
                                                'ymin': y_min,
                                                'xmax': x_max,
                                                'ymax': y_max
                                            }
                                        })
                                    
                                    example['pascal_voc_annotation'] = {
                                        'filename': file_id,
                                        'size': {
                                            'width': width,
                                            'height': height,
                                            'depth': 3
                                        },
                                        'objects': pascal_objects
                                    }
                            
                        examples.append(example)
                    
                    # Create dataset for this split if we have examples
                    if examples:
                        # Define features based on what's included
                        features = Features({
                            'image': Image(),
                            'width': Value('int64'),
                            'height': Value('int64')
                        })
                        
                        # Add additional features based on what's included
                        if self.include_yolo:
                            features['yolo_annotation'] = yolo_features()
                        
                        if self.include_coco:
                            features['coco_annotation'] = coco_annotation_features()
                        
                        if self.include_pascal_voc:
                            features['pascal_voc_annotation'] = pascal_voc_features()
                        
                        # Create the dataset
                        partition_dict[split] = Dataset.from_list(examples, features=features)
                        print(f"  Created {split} dataset for partition {partition_name} with {len(examples)} examples")
                    else:
                        print(f"  Warning: No examples found for split {split} in partition {partition_name}")
                
                # Add the partition dataset if we have any splits
                if partition_dict:
                    partition_datasets[partition_name] = DatasetDict(partition_dict)
                
            except Exception as e:
                print(f"Error processing partition {partition_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return partition_datasets
    
    def upload(self, dataset_dict: Optional[DatasetDict] = None) -> None:
        """
        Upload the dataset to Hugging Face.
        
        Args:
            dataset_dict: Optional pre-prepared dataset dictionary
        """
        # Prepare dataset if not provided
        if dataset_dict is None:
            dataset_dict = self.prepare_dataset()
        
        if not dataset_dict:
            print("Error: No datasets to upload. Please check your data directory and format.")
            return
            
        # Create partition datasets if partition directory is specified
        partition_datasets = {}
        if self.partition_dir:
            partition_datasets = self.create_partition_datasets()
            
        # Create repository if it doesn't exist
        try:
            self.api.create_repo(
                repo_id=self.dataset_name,
                repo_type="dataset",
                private=self.private,
                exist_ok=True
            )
            print(f"Repository {self.dataset_name} created or already exists.")
        except Exception as e:
            print(f"Error creating repository: {e}")
            return
        
        # Push dataset to Hub
        try:
            dataset_dict.push_to_hub(
                self.dataset_name,
                token=self.token,
                private=self.private
            )
            print(f"Successfully uploaded dataset to {self.dataset_name}")
            
            # Create and upload a README file
            format_name = "COCO" if self.format == "coco" else "YOLO"
            splits = list(dataset_dict.keys())
            
            # Get current date for metadata
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Get class names for object detection
            class_names_list = []
            if self.format == "yolo" and self.config_file:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    class_names_list = config.get('names', [])
            
            # Get the basename of the dataset
            dataset_basename = os.path.basename(self.dataset_name)
            
            # Create basic YAML frontmatter
            readme_content = f"""---
annotations_creators:
- expert-generated
language_creators:
- found
language:
- en
license:
- cc-by-4.0
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- object-detection
task_ids:
- instance-segmentation
pretty_name: {dataset_basename}
---

# {dataset_basename} Dataset

An object detection dataset in {format_name} format containing {len(splits)} splits: {', '.join(splits)}.

## Dataset Metadata

* **License:** CC-BY-4.0 (Creative Commons Attribution 4.0 International)
* **Version:** 1.0
* **Date Published:** {current_date}
* **Cite As:** TBD

## Dataset Details

- Format: {format_name}
- Splits: {', '.join(splits)}
"""
            
            if class_names_list:
                readme_content += f"\n- Classes: {', '.join(class_names_list)}\n"
            
            # Add information about additional formats
            additional_formats = []
            if self.include_coco and self.format != "coco":
                additional_formats.append("COCO format annotations")
            if self.include_pascal_voc:
                additional_formats.append("Pascal VOC format annotations")
            if self.include_yolo and self.format != "yolo":
                additional_formats.append("YOLO format annotations")
                
            if additional_formats:
                readme_content += "\n## Additional Formats\n\n"
                for fmt in additional_formats:
                    readme_content += f"- Includes {fmt}\n"
            
            readme_content += "\n\n## Usage\n\n```python\nfrom datasets import load_dataset\n\ndataset = load_dataset(\""
            readme_content += self.dataset_name
            readme_content += "\")\n```\n"
            
            with open("README.md", "w") as f:
                f.write(readme_content)
            
            self.api.upload_file(
                path_or_fileobj="README.md",
                path_in_repo="README.md",
                repo_id=self.dataset_name,
                repo_type="dataset"
            )
            print("Added dataset card to the repository.")
            
            # Clean up temporary files
            if os.path.exists("README.md"):
                os.remove("README.md")
                
        except Exception as e:
            print(f"Error uploading dataset: {e}")
            
        # Upload partition datasets if available
        if partition_datasets:
            print("\nUploading partition datasets to the main dataset...")
            
            # Create a combined dataset dictionary with all partitions
            # We need to create a new DatasetDict, not just copy the dictionary
            combined_dataset_dict = DatasetDict()
            
            # First add the original splits
            for split_name, split_dataset in dataset_dict.items():
                combined_dataset_dict[split_name] = split_dataset
            
            # Add each partition as a new split in the main dataset
            for partition_name, partition_dict in partition_datasets.items():
                print(f"Processing partition: {partition_name}")
                
                try:
                    # Combine all examples from all splits in this partition into a single dataset
                    all_examples = []
                    total_examples = 0
                    
                    # Collect examples from each split
                    for split_name, split_dataset in partition_dict.items():
                        # Convert the dataset to a list of examples
                        examples = list(split_dataset)
                        all_examples.extend(examples)
                        total_examples += len(examples)
                    
                    # If we have examples, create a single dataset for this partition
                    if all_examples:
                        # Get features from one of the splits
                        features = next(iter(partition_dict.values())).features
                        
                        # Create a new dataset with all examples
                        combined_dataset = Dataset.from_list(all_examples, features=features)
                        
                        # Add this as a single split named after the partition
                        combined_dataset_dict[partition_name] = combined_dataset
                        print(f"  Added {partition_name} split with {total_examples} examples")
                        
                except Exception as e:
                    print(f"Error processing partition {partition_name}: {e}")
            
            # Push the combined dataset to Hub
            try:
                print("Uploading combined dataset with partitions...")
                combined_dataset_dict.push_to_hub(
                    self.dataset_name,
                    token=self.token,
                    private=self.private
                )
                print(f"Successfully uploaded combined dataset with partitions to {self.dataset_name}")
                
                # Update the README to include information about partitions
                format_name = "COCO" if self.format == "coco" else "YOLO"
                splits = list(combined_dataset_dict.keys())
                
                # Get current date for metadata
                from datetime import datetime
                current_date = datetime.now().strftime("%Y-%m-%d")
                
                # Get the basename of the dataset
                dataset_basename = os.path.basename(self.dataset_name)
                
                # Create basic YAML frontmatter
                readme_content = f"""---
annotations_creators:
- expert-generated
language_creators:
- found
language:
- en
license:
- cc-by-4.0
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- object-detection
task_ids:
- instance-segmentation
pretty_name: {dataset_basename}
---

# {dataset_basename} Dataset

An object detection dataset in {format_name} format containing {len(splits)} splits: {', '.join(splits)}.

## Dataset Metadata

* **License:** CC-BY-4.0 (Creative Commons Attribution 4.0 International)
* **Version:** 1.0
* **Date Published:** {current_date}
* **Cite As:** TBD

## Dataset Details

- Format: {format_name}
- Splits: {', '.join(splits)}

## Partitions

This dataset includes the following partitions:
"""
                
                # Add information about partitions
                for partition_name in partition_datasets.keys():
                    readme_content += f"- {partition_name}: Available as a single split named '{partition_name}'\n"
                
                # Add information about additional formats
                additional_formats = []
                if self.include_coco and self.format != "coco":
                    additional_formats.append("COCO format annotations")
                if self.include_pascal_voc:
                    additional_formats.append("Pascal VOC format annotations")
                if self.include_yolo and self.format != "yolo":
                    additional_formats.append("YOLO format annotations")
                    
                if additional_formats:
                    readme_content += "\n## Additional Formats\n\n"
                    for fmt in additional_formats:
                        readme_content += f"- Includes {fmt}\n"
                
                readme_content += "\n\n## Usage\n\n```python\nfrom datasets import load_dataset\n\n# Load the main dataset\ndataset = load_dataset(\""
                readme_content += self.dataset_name
                readme_content += "\")\n\n# Load a specific partition\npartition_dataset = load_dataset(\""
                readme_content += self.dataset_name
                readme_content += "\", \"0.05x_partition\")\n```\n"
                
                with open("README.md", "w") as f:
                    f.write(readme_content)
                
                self.api.upload_file(
                    path_or_fileobj="README.md",
                    path_in_repo="README.md",
                    repo_id=self.dataset_name,
                    repo_type="dataset"
                )
                print("Updated dataset card with partition information.")
                
                # Clean up temporary files
                if os.path.exists("README.md"):
                    os.remove("README.md")
                    
            except Exception as e:
                print(f"Error uploading combined dataset: {e}")

def main():
    """Command-line interface for the Object Detection Uploader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload object detection datasets to Hugging Face")
    
    # Required arguments
    parser.add_argument("--data_dir", required=True, 
                       help="Path to data directory containing images and annotations")
    parser.add_argument("--dataset_name", required=True, 
                       help="Name for dataset on Hugging Face (username/dataset-name)")
    
    # Optional arguments
    parser.add_argument("--token", help="Hugging Face API token (will use stored token if not provided)")
    parser.add_argument("--format", choices=["coco", "yolo"], default="coco",
                       help="Format of annotations (default: coco)")
    parser.add_argument("--config_file", help="Path to configuration file for YOLO format")
    parser.add_argument("--partition-dir", help="Path to partition directory")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    parser.add_argument("--include_pascal_voc", action="store_true", 
                       help="Include Pascal VOC format annotations as a column")
    parser.add_argument("--include_coco", action="store_true", 
                       help="Include COCO format annotations as a column")
    parser.add_argument("--include_yolo", action="store_true", 
                       help="Include YOLO format annotations as a column")
    
    args = parser.parse_args()
    
    # Create uploader and upload dataset
    uploader = ObjectDetectionUploader(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        token=args.token,
        format=args.format,
        config_file=args.config_file,
        partition_dir=args.partition_dir,
        private=args.private,
        include_pascal_voc=args.include_pascal_voc,
        include_coco=args.include_coco,
        include_yolo=args.include_yolo
    )
    
    uploader.upload()

if __name__ == "__main__":
    main() 
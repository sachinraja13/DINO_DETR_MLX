import os
import json
import numpy as np
import mlx.core as mx
import random


class SyntheticDataset:
    def __init__(self, num_samples=100, num_classes=91):
        self.ids = list(range(num_samples))
        self.num_classes = num_classes
        self.num_samples = num_samples
        
    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return self
    
    def generate_bounding_boxes(self, n):
        w = np.random.uniform(0, 1, size=n)
        h = np.random.uniform(0, 1, size=n)
        x = np.random.uniform(0, 1 - w)
        y = np.random.uniform(0, 1 - h)
        boxes = np.stack([x, y, w, h], axis=1)
        return boxes

    def __getitem__(self, index):
        batch_size = 2
        height, width = 480, 640
        image = mx.array(np.random.randn(height, width, 3).astype(np.float32))
        max_targets_per_image = 50
        num_targets = np.random.randint(1, max_targets_per_image + 1)
        labels_np = np.random.randint(0, self.num_classes, size=(num_targets,), dtype=np.int32)
        boxes_np = self.generate_bounding_boxes(num_targets).astype(np.float32)
        target = {
            'labels': mx.array(labels_np),
            'boxes': mx.array(boxes_np)
        }
        return image, target

def build(image_set, args):
    dataset = SyntheticDataset(num_samples=args.num_samples_synthetic_dataset, num_classes=args.num_classes_synthetic_dataset)
    return dataset
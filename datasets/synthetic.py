import os
import json
import numpy as np
import mlx.core as mx
import random


class SyntheticDataset:
    def __init__(
        self, 
        num_samples=100,
        num_classes=91, 
        precision='full', 
        min_targets_per_image=10, 
        max_targets_per_image=100,
        pad_labels_to_n_max_ground_truth=False,
        n_max_ground_truth=800
    ):
        self.ids = list(range(num_samples))
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.precision = precision
        self.min_targets_per_image = min_targets_per_image
        self.max_targets_per_image = max_targets_per_image
        self.pad_labels_to_n_max_ground_truth = pad_labels_to_n_max_ground_truth
        self.n_max_ground_truth = n_max_ground_truth
        
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
        image = mx.array(np.random.randn(height, width, 3))
        if self.precision == 'half':
            image = image.astype(mx.float16)
        
        num_targets = np.random.randint(self.min_targets_per_image, self.max_targets_per_image + 1)
        labels_np = np.random.randint(0, self.num_classes, size=(num_targets,), dtype=np.int16)
        boxes_np = self.generate_bounding_boxes(num_targets)
        num_objects = num_targets
        pad_size = self.n_max_ground_truth - num_objects
        # if self.pad_labels_to_n_max_ground_truth:
        #     if pad_size >= 0:
        #         labels_np = np.pad(labels_np, ((0, pad_size)))
        #         boxes_np = np.pad(boxes_np, ((0, pad_size), (0, 0)))
        #     else:
        #         labels_np = labels_np[0 : self.n_max_ground_truth]
        #         boxes_np = boxes_np[0 : self.n_max_ground_truth]

        if self.precision == 'half':
            target = {
                'labels': mx.array(labels_np).astype(mx.int16),
                'boxes': mx.array(boxes_np).astype(mx.float16),
                'num_objects' : num_objects,
                'size' : mx.array([int(height), int(width)]),
                'orig_size' : mx.array([int(height), int(width)])
            }
        else:
            target = {
                'labels': mx.array(labels_np).astype(mx.int32),
                'boxes': mx.array(boxes_np).astype(mx.float32),
                'num_objects' : num_objects,
                'size' : mx.array([int(height), int(width)]),
                'orig_size' : mx.array([int(height), int(width)])
            }
        return image, target

def build(image_set, args):
    dataset = SyntheticDataset(
        num_samples=args.num_samples_synthetic_dataset, 
        num_classes=args.num_classes_synthetic_dataset, 
        precision=args.precision,
        min_targets_per_image=args.min_targets_per_image,
        max_targets_per_image=args.max_targets_per_image,
        pad_labels_to_n_max_ground_truth=args.pad_labels_to_n_max_ground_truth,
        n_max_ground_truth=args.n_max_ground_truth
        )
    return dataset
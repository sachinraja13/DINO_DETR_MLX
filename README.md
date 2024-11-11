# DINO DETR - Apple MLX Port

This repository is an adaptation of the **DINO DETR** ([here](https://github.com/IDEA-Research/DINO)) object detection model for **Apple's MLX** platform, enabling efficient and scalable deployment of advanced object detection models on Apple hardware. DINO DETR is a state-of-the-art object detection model that builds upon DETR by introducing deformable attention mechanisms and better query matching strategies. For the time being, distributed training is not supported. The ported model also does not support segmentation as of now.

## Table of Contents

- [News](#news)
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## News

- *Stable Matching and Stable DINO Loss Criterion loss is now supported.*
- *Results on DINO SWIN-L on COCO Val 2017 dataset are added.*
- *Added jupyter notebooks for inference and visualization with both ResNet and SWIN-L backbones.*

## Overview

DINO DETR (DEtection TRansformer with Improved Matching) pushes the boundaries of object detection by combining transformer architectures with a deformable attention mechanism, enhancing the model's performance on small objects and faster convergence. This port leverages **Apple's MLX** framework to provide optimized performance on Apple Silicon devices, making it easier to deploy on iOS/macOS apps.

## Features

- **Deformable Attention**: Efficient handling of multi-scale features for better performance on small objects.
- **Improved Query Matching**: Enhanced matching algorithm between predicted and ground truth boxes for better accuracy.
- **Apple MLX Integration**: Full support for Apple Silicon devices, including M1, M2, and future chips.
- **Optimized Performance**: Significant performance improvements when running on Apple hardware, thanks to MLX framework and Metal optimizations. Special mention to @awnihannun, @barronalex and @angeloskath.

## Installation

```
git clone https://github.com/sachinraja13/DINO_DETR_MLX.git
git lfs fetch --all
```

### Prerequisites

- macOS 12 or later
- Xcode 13 or later
- Apple Silicon (M1 or later) or Intel Mac with macOS
- Python 3.9 or later

### Usage

To load pretrained pytorch weights from the original authors, set the *load_pytorch_weights* flag in the config file to True and *pytorch_weights_path* with the path of pytorch weights pth file.

For testing with synthetic data use *DINO_4scale_synthetic.py* as the config file and for coco, use *DINO_4scale_coco2017.py*

For Training:

```
export coco_path = 'path/to/coco/dataset'
python main.py \
 --output_dir logs/DINO/R50-MS4 -c config/DINO/DINO_4scale_coco2017.py --coco_path $coco_path \
 --options embed_init_tgt=TRUE
```

For evaluation:

```
python main.py \
 --output_dir logs/DINO/R50-MS4-%j \
 -c config/DINO/DINO_4scale.py --coco_path $coco_path  \
 --eval --resume $checkpoint \
 --options embed_init_tgt=TRUE
```

### Results

Results on COCO 2017 Validation dataset when evaluated using preloaded pytorch weights on ResNet50 Backbone:

```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.477
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.665
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.517
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.310
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.515
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.619
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.369
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.633
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.703
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.749
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.863
```

Results on COCO 2017 Validation dataset when evaluated using preloaded pytorch weights on SWIN-L Backbone:

```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.554
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.750
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.602
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.596
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.721
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.403
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.680
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.747
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.595
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.791
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.899
```

### Contributing

Contributions are welcome! Please open a pull request or issue to suggest improvements or report bugs.

### Licence

DINO is released under the Apache 2.0 license. Please see the LICENSE file for more information.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

### Citations

DN-DETR: Accelerate DETR Training by Introducing Query DeNoising.
Feng Li*, Hao Zhang*, Shilong Liu, Jian Guo, Lionel M. Ni, Lei Zhang.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2022.

DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR.
Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, Lei Zhang.
International Conference on Learning Representations (ICLR) 2022.

DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection.
Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum
arXiv 2203.03605

Thanks to mlx-image for resnet and swin ports to MLX.

If you find this work useful, please cite this repository and the original authors of DINO DETR as follows:

```
@misc{zhang2022dino,
      title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection}, 
      author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2203.03605},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{li2022dn,
      title={Dn-detr: Accelerate detr training by introducing query denoising},
      author={Li, Feng and Zhang, Hao and Liu, Shilong and Guo, Jian and Ni, Lionel M and Zhang, Lei},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={13619--13627},
      year={2022}
}

@inproceedings{
      liu2022dabdetr,
      title={{DAB}-{DETR}: Dynamic Anchor Boxes are Better Queries for {DETR}},
      author={Shilong Liu and Feng Li and Hao Zhang and Xiao Yang and Xianbiao Qi and Hang Su and Jun Zhu and Lei Zhang},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=oMI9PjOb9Jl}
}
```

"""
Inference Mask R-CNN

Edited by Lee, Gimun
Origin on https://github.com/matterport/Mask_RCNN/blob/master/samples/demo.ipynb
"""
from mask_rcnn.mrcnn import utils
import mask_rcnn.mrcnn.model as modellib
from mask_rcnn.mrcnn import visualize
import mask_rcnn.mrcnn.config as mrcnn_config

import os
from PIL import Image

import numpy as np

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join('.', 'mask_rcnn_coco.h5')
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(mrcnn_config.Config):
    """Configuration for training on MS COCO.
        Derives from the base Config class and overrides values specific
        to the COCO dataset.
        """
    # Give the configuration a recognizable name
    NAME = "coco"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


CONFIG = InferenceConfig()
CONFIG.display()

MODEL = modellib.MaskRCNN(mode='inference', config=CONFIG, model_dir='temp')
MODEL.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def inference_then_save(image: Image.Image, save_path: str) -> Image.Image:
    array = np.array(image)

    detected_results = MODEL.detect([array], verbose=0)
    detected_results = detected_results[0]
    visualized = visualize.display_instances(array, detected_results['rois'], detected_results['masks'], detected_results['class_ids'], CLASS_NAMES,
                                             detected_results['scores'], save_path=save_path)

    return Image.fromarray(visualized)


def main():
    image = Image.open('whitney.jpg')
    visualized = inference_then_save(image, 'whitney_segmented.jpg')


if __name__ == '__main__':
    main()

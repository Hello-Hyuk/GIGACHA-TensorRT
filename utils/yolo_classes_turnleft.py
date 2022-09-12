"""yolo_classes.py
NOTE: Number of YOLO COCO output classes differs from SSD COCO models.
"""

COCO_CLASSES_LIST = [
    'turn_left_sign',
    
]

# For translating YOLO class ids (0~79) to SSD class ids (0~90)
yolo_cls_to_ssd = [
    0, 
]

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == 1:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
    else:
        return {i: 'CLS%d' % i for i in range(category_num)}

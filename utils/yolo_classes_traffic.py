"""yolo_classes.py
NOTE: Number of YOLO COCO output classes differs from SSD COCO models.
"""

COCO_CLASSES_LIST = [
    'Red_3',
    'Red_4',
    'Yellow_3',
    'Yellow_4',
    'Green_3',
    'Green_4',
    'Red_Left_4',
    'Green_Left_4',
]

# For translating YOLO class ids (0~79) to SSD class ids (0~90)
yolo_cls_to_ssd = [
    0, 1, 2, 3, 4, 5, 6, 7,
]

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == 8:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
    else:
        return {i: 'CLS%d' % i for i in range(category_num)}

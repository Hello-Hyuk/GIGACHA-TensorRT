"""yolo_classes.py

NOTE: Number of YOLO COCO output classes differs from SSD COCO models.
"""

COCO_CLASSES_LIST = [
    'a1',
    'a2',
    'a3',
    'b1',
    'b2',
    'b3'
]

# For translating YOLO class ids (0~79) to SSD class ids (0~90)
#yolo_cls_to_ssd = [
#    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
#    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
#    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
#    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
#    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
#]
yolo_cls_to_ssd = [
        0,1,2,3,4,5
        ]

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num ==6: # 80:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
    else:
        return {i: 'CLS%d' % i for i in range(category_num)}

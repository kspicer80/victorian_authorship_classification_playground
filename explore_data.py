import numpy as np

def get_num_classes(labels):
    missing_authors = [5, 7, 31, 47, 49]
    num_classes = max(labels) + 1
    missing_classes = [item for item in range(num_classes) if item not in missing_authors]
    if len(missing_classes):
       raise ValueError()
    if num_classes <= 1:
        raise ValueError('Invalid number of labels: {num_classes}.'
                         'Please make sure there are at least two classes '
                         'of samples'.format(num_classes=num_classes))
    return num_classes
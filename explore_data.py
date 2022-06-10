import numpy as np

def get_num_classes(labels):
    num_classes = set(labels)
    missing_classes = [i for i in num_classes if i not in labels]
    if len(missing_classes):
       raise ValueError()
    if num_classes <= 1:
        raise ValueError('Invalid number of labels: {num_classes}.'
                         'Please make sure there are at least two classes '
                         'of samples'.format(num_classes=num_classes))
    return num_classes 
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset


class ErrorDataset(Dataset):
    """Dataset for error prediction training."""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class ErrorPredictor(nn.Module):
    """Neural network to predict error terms.
    
    Output dimensions (8 total):
        0: distal error (radial position)
        1: perpendicular error (lateral position)
        2: height error (z position)
        3: yaw error (orientation)
        4: width error (box dimension)
        5: length error (box dimension)
        6: box_height error (box dimension)
        7: missed_rate (detection miss probability)
    """
    def __init__(self, input_dim, hidden_dims, output_dim=8):
        super(ErrorPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def extract_features_from_detection(
    gt_box, det_box, distance, class_name, CLASSES
):
    """Extract features for error prediction."""
    features = []
    
    gt_center = gt_box[:3]
    gt_dims = gt_box[3:6]

    # Distance features (remains)
    features.extend([distance, distance**2, distance**3])
    
    # GT Size features (remains)
    features.extend([gt_dims[0], gt_dims[1], gt_dims[2]])
    
    # GT Position features (remains)
    features.extend([gt_center[0], gt_center[1], gt_center[2]])
        
    # Remove leaky features if detection is present
    if det_box is not None:  # Valid detection
        pass  # Intentionally no extra features from detection
        
    else:  # Missed detection
        pass  # No change needed here

    # Class encoding (remains)
    class_encoding = {cls: [0]*len(CLASSES) for cls in CLASSES}
    if class_name in class_encoding:
        class_idx = CLASSES.index(class_name)
        class_encoding[class_name][class_idx] = 1

    encoding = class_encoding.get(class_name, [0] * len(CLASSES))
    features.extend(encoding)
    
    return np.array(features) 

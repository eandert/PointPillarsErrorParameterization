import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import pickle
from tqdm import tqdm

from models import (
    ErrorPredictor, ErrorDataset, extract_features_from_detection
)
from pointpillars.dataset import Kitti
from pointpillars.model import PointPillars
from evaluate_errors import ErrorAnalyzer


def prepare_training_data(data_root, ckpt_path, classes, max_samples, iou_thresh):
    """Prepare training data for error prediction."""
    print("Preparing training data...")
    train_dataset = Kitti(data_root=data_root, split='train')

    model = PointPillars(nclasses=len(classes)).cuda()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    error_analyzer = ErrorAnalyzer()
    
    features_list, targets_list, distances_list = [], [], []
    
    sample_indices = range(len(train_dataset))
    if max_samples > 0:
        sample_indices = range(min(max_samples, len(train_dataset)))
        print(f"Processing {len(sample_indices)} samples out of "
              f"{len(train_dataset)} total")

    with torch.no_grad():
        for i in tqdm(sample_indices):
            data_dict = train_dataset[i]
            
            pts = torch.from_numpy(data_dict['pts']).cuda()
            gt_bboxes_3d = torch.from_numpy(
                data_dict['gt_bboxes_3d']).cuda()
            gt_labels = torch.from_numpy(
                data_dict['gt_labels']).cuda()
            
            batch_results = model(
                batched_pts=[pts],
                mode='val',
                batched_gt_bboxes=[gt_bboxes_3d],
                batched_gt_labels=[gt_labels]
            )
            
            del pts, gt_bboxes_3d, gt_labels
            torch.cuda.empty_cache()

            calib_info = data_dict['calib_info']
            tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
            r0_rect = calib_info['R0_rect'].astype(np.float32)
            P2 = calib_info['P2'].astype(np.float32)
            image_shape = data_dict['image_info']['image_shape']
            
            from pointpillars.utils import (
                keep_bbox_from_image_range, keep_bbox_from_lidar_range
            )
            pcd_limit_range = np.array(
                [0, -40, -3, 70.4, 40, 0.0], dtype=np.float32
            )
            
            result = batch_results[0]
            result_filter = keep_bbox_from_image_range(
                result, tr_velo_to_cam, r0_rect, P2, image_shape
            )
            result_filter = keep_bbox_from_lidar_range(
                result_filter, pcd_limit_range
            )

            lidar_bboxes = result_filter['lidar_bboxes']
            labels, scores = result_filter['labels'], result_filter['scores']
            
            format_result = {
                'name': [], 'location': [], 'dimensions': [], 
                'rotation_y': [], 'score': []
            }
            LABEL2CLASSES = {i: k for i, k in enumerate(classes)}
            for label, score, lidar_bbox in zip(labels, scores, lidar_bboxes):
                format_result['name'].append(LABEL2CLASSES[label])
                format_result['location'].append(lidar_bbox[:3])
                format_result['dimensions'].append(lidar_bbox[3:6])
                format_result['rotation_y'].append(lidar_bbox[6])
                format_result['score'].append(score)
            
            for key in format_result:
                format_result[key] = np.array(format_result[key])

            gt_annos = train_dataset.data_infos[train_dataset.sorted_ids[i]]['annos']

            for class_name in classes:
                gt_mask = gt_annos['name'] == class_name
                det_mask = format_result['name'] == class_name
                if not gt_mask.any():
                    continue

                gt_boxes = np.concatenate([
                    gt_annos['location'][gt_mask],
                    gt_annos['dimensions'][gt_mask],
                    gt_annos['rotation_y'][gt_mask][:, None]
                ], axis=1)

                det_boxes = np.empty((0, 7))
                if np.any(det_mask):
                    det_boxes = np.concatenate([
                        format_result['location'][det_mask],
                        format_result['dimensions'][det_mask],
                        format_result['rotation_y'][det_mask][:, None]
                    ], axis=1)

                gt_distances = error_analyzer.calculate_distance(
                    gt_boxes[:, :3]
                )
                gt_to_det, _, _ = error_analyzer.match_detections(
                    gt_boxes, det_boxes, iou_thresh=iou_thresh
                )

                for gt_idx, gt_box in enumerate(gt_boxes):
                    distance = gt_distances[gt_idx]
                    det_box = None
                    if gt_idx in gt_to_det:
                        det_box = det_boxes[gt_to_det[gt_idx]]
                    
                    distal, perp, height = (np.nan, np.nan, np.nan)
                    missed_rate = 1.0
                    if det_box is not None:
                        distal, perp, height = error_analyzer.compute_errors(
                            gt_box, det_box
                        )
                        missed_rate = 0.0

                    features = extract_features_from_detection(
                        gt_box, det_box, distance, class_name, classes
                    )
                    targets = [
                        distal if not np.isnan(distal) else 0.0,
                        perp if not np.isnan(perp) else 0.0,
                        height if not np.isnan(height) else 0.0,
                        missed_rate
                    ]
                    features_list.append(features)
                    targets_list.append(targets)
                    distances_list.append(distance)

    return (np.array(features_list), np.array(targets_list), 
            np.array(distances_list))


def train_error_predictor(features, targets, hidden_dims, learning_rate,
                          batch_size, epochs, output_dir):
    """Train the error prediction neural network."""
    print("Training neural network error predictor...")
    X_train, X_val, y_train, y_val = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    train_dataset = ErrorDataset(X_train_scaled, y_train)
    val_dataset = ErrorDataset(X_val_scaled, y_val)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ErrorPredictor(
        input_dim=features.shape[1], hidden_dims=hidden_dims
    ).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.cuda(), batch_targets.cuda()
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features, batch_targets = batch_features.cuda(), batch_targets.cuda()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: '
                  f'{train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

    # Save model and scaler
    torch.save(model.state_dict(), os.path.join(output_dir, 'error_predictor.pth'))
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Neural network model and scaler saved.")
    return model, scaler


def train_linear_models(distances, targets, output_dir):
    """Train linear and quadratic models for comparison."""
    print("Training linear and quadratic models...")
    X = distances.reshape(-1, 1)
    
    linear_models = {}
    for i, target_name in enumerate(['distal', 'perp', 'height', 'missed_rate']):
        valid_mask = ~np.isnan(targets[:, i])
        X_valid, y_valid = X[valid_mask], targets[valid_mask, i]

        if X_valid.shape[0] == 0:
            print(f"Skipping {target_name} model, no valid data.")
            continue
            
        # Linear
        linear_model = LinearRegression()
        linear_model.fit(X_valid, y_valid)
        linear_models[f'{target_name}_linear'] = linear_model
        
        # Quadratic
        quadratic_model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ])
        quadratic_model.fit(X_valid, y_valid)
        linear_models[f'{target_name}_quadratic'] = quadratic_model
    
    with open(os.path.join(output_dir, 'linear_models.pkl'), 'wb') as f:
        pickle.dump(linear_models, f)
    print("Linear/quadratic models saved.")
    return linear_models


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare training data
    features, targets, distances = prepare_training_data(
        args.data_root, args.ckpt, args.classes, args.max_samples, args.iou_thresh
    )
    
    # Train NN predictor
    train_error_predictor(
        features, targets, args.hidden_dims, args.learning_rate, 
        args.batch_size, args.epochs, args.output_dir
    )
    
    # Train linear models with the correct 'distances' variable
    train_linear_models(distances, targets, args.output_dir)
    
    print("\n🎉 All models trained successfully!")
    print(f"Models and scalers saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train All Error Predictor Models'
    )
    parser.add_argument(
        '--data_root', default='./data/kitti', help='Path to KITTI dataset'
    )
    parser.add_argument(
        '--ckpt', default='pretrained/epoch_160.pth', 
        help='Path to PointPillars checkpoint'
    )
    parser.add_argument(
        '--output_dir', default='error_predictor_results', 
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--max_samples', type=int, default=3000, 
        help='Max training samples for data generation (-1 for all)'
    )
    parser.add_argument(
        '--hidden_dims', nargs='+', type=int, default=[128, 64, 32], 
        help='NN hidden layer dimensions'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.001, help='Learning rate'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64, help='Batch size'
    )
    parser.add_argument(
        '--epochs', type=int, default=100, help='Number of epochs'
    )
    parser.add_argument(
        '--iou_thresh', type=float, default=0.3, help='IoU threshold for matching'
    )
    parser.add_argument(
        '--classes', nargs='+', default=['Car', 'Pedestrian', 'Cyclist'], 
        help='Classes to process'
    )
    
    args = parser.parse_args()
    main(args) 

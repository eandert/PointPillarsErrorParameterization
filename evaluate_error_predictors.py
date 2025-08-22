import argparse
import numpy as np
import os
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from models import ErrorPredictor
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars
from evaluate_errors import ErrorAnalyzer
from models import extract_features_from_detection


def load_trained_nn_model(model_path, scaler_path, input_dim, hidden_dims):
    """Load trained neural network model."""
    model = ErrorPredictor(input_dim=input_dim, hidden_dims=hidden_dims).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler


def load_linear_models(model_path):
    """Load trained linear/quadratic models."""
    with open(model_path, 'rb') as f:
        linear_models = pickle.load(f)
    return linear_models


def prepare_evaluation_data(data_root, ckpt_path, CLASSES, iou_thresh):
    """Prepare evaluation data (validation set)."""
    print("Preparing evaluation data...")
    
    # Load validation dataset
    val_dataset = Kitti(data_root=data_root, split='val')
    val_dataloader = get_dataloader(
        dataset=val_dataset, 
        batch_size=1, 
        num_workers=0,
        shuffle=False
    )
    
    # Load trained model
    model = PointPillars(nclasses=len(CLASSES)).cuda()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    
    # Initialize error analyzer
    error_analyzer = ErrorAnalyzer()
    
    features_list = []
    targets_list = []
    distances_list = []
    class_names_list = []
    
    print("Running inference on validation data...")
    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(val_dataloader)):
            # Move data to GPU
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].cuda()
            
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            
            # Run inference
            batch_results = model(
                batched_pts=batched_pts, 
                mode='val',
                batched_gt_bboxes=batched_gt_bboxes, 
                batched_gt_labels=batched_labels
            )
            
            # Process each sample in batch
            for j, result in enumerate(batch_results):
                gt_result = data_dict['batched_gt_bboxes'][j]
                gt_labels = data_dict['batched_labels'][j]
                
                # Process model output similar to evaluate_errors.py
                calib_info = data_dict['batched_calib_info'][j]
                tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
                r0_rect = calib_info['R0_rect'].astype(np.float32)
                P2 = calib_info['P2'].astype(np.float32)
                image_shape = data_dict['batched_img_info'][j]['image_shape']
                
                # Filter results
                from pointpillars.utils import (
                    keep_bbox_from_image_range, 
                    keep_bbox_from_lidar_range
                )
                pcd_limit_range = np.array(
                    [0, -40, -3, 70.4, 40, 0.0], dtype=np.float32
                )
                
                result_filter = keep_bbox_from_image_range(
                    result, tr_velo_to_cam, r0_rect, P2, image_shape
                )
                result_filter = keep_bbox_from_lidar_range(
                    result_filter, pcd_limit_range
                )
                
                lidar_bboxes = result_filter['lidar_bboxes']
                labels, scores = result_filter['labels'], result_filter['scores']
                bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
                
                # Create formatted result
                format_result = {
                    'name': [],
                    'location': [],
                    'dimensions': [],
                    'rotation_y': [],
                    'score': []
                }
                
                LABEL2CLASSES = {i: k for i, k in enumerate(CLASSES)}
                for lidar_bbox, label, score, bbox2d, camera_bbox in \
                    zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
                    format_result['name'].append(LABEL2CLASSES[label])
                    format_result['location'].append(camera_bbox[:3])
                    format_result['dimensions'].append(camera_bbox[3:6])
                    format_result['rotation_y'].append(camera_bbox[6])
                    format_result['score'].append(score)
                
                # Convert to numpy arrays
                for key in format_result:
                    format_result[key] = np.array(format_result[key])
                
                for class_name in CLASSES:
                    gt_mask = gt_labels == CLASSES.index(class_name)
                    det_mask = format_result['name'] == class_name
                    
                    if not gt_mask.any():
                        continue
                    
                    # Format boxes for IoU calculation
                    gt_masked_cpu = gt_result[gt_mask].cpu().numpy()
                    gt_boxes = np.concatenate([
                        gt_masked_cpu[:, :3],  # location
                        gt_masked_cpu[:, 3:6][:, [1, 0, 2]],  # w,l,h
                        gt_masked_cpu[:, 6:7]   # rotation
                    ], axis=1)
                    
                    if np.any(det_mask):
                        det_boxes = np.concatenate([
                            format_result['location'][det_mask],
                            format_result['dimensions'][det_mask][:, [1, 0, 2]],
                            format_result['rotation_y'][det_mask][:, None]
                        ], axis=1)
                    else:
                        det_boxes = np.empty((0, 7))
                    
                    # Calculate distances
                    gt_distances = error_analyzer.calculate_distance(gt_masked_cpu[:, :3])
                    
                    # Match detections to ground truth
                    gt_to_det, unmatched_gt, _ = error_analyzer.match_detections(
                        gt_boxes, det_boxes, iou_thresh=iou_thresh
                    )
                    
                    # Extract features and targets for each ground truth object
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        distance = gt_distances[gt_idx]
                        
                        if gt_idx in gt_to_det:
                            # Valid detection
                            det_box = det_boxes[gt_to_det[gt_idx]]
                            distal, perp, height = error_analyzer.compute_errors(gt_box, det_box)
                            missed_rate = 0.0
                        else:
                            # Missed detection
                            det_box = None
                            distal, perp, height = np.nan, np.nan, np.nan
                            missed_rate = 1.0
                        
                        # Extract features
                        features = extract_features_from_detection(
                            gt_box, det_box, distance, class_name, CLASSES
                        )
                        
                        # Prepare targets
                        targets = [
                            distal if not np.isnan(distal) else 0.0,
                            perp if not np.isnan(perp) else 0.0,
                            height if not np.isnan(height) else 0.0,
                            missed_rate
                        ]
                        
                        features_list.append(features)
                        targets_list.append(targets)
                        distances_list.append(distance)
                        class_names_list.append(class_name)
    
    return (np.array(features_list), np.array(targets_list), 
            np.array(distances_list), class_names_list)


def evaluate_predictions_nn(model, scaler, features):
    """Evaluate neural network predictions."""
    print("Evaluating neural network predictions...")
    model.eval()
    with torch.no_grad():
        features_scaled = scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).cuda()
        predictions = model(features_tensor).cpu().numpy()
    return predictions


def evaluate_predictions_linear(linear_models, distances, targets):
    """Evaluate linear/quadratic model predictions."""
    print("Evaluating linear/quadratic model predictions...")
    predictions = np.zeros_like(targets)
    X = distances.reshape(-1, 1)
    
    for i, target_name in enumerate(['distal', 'perp', 'height', 'missed_rate']):
        # Prioritize quadratic model if available
        quad_model = linear_models.get(f'{target_name}_quadratic')
        linear_model = linear_models.get(f'{target_name}_linear')

        if quad_model:
            predictions[:, i] = quad_model.predict(X)
        elif linear_model:
            predictions[:, i] = linear_model.predict(X)

    return predictions


def calculate_metrics(predictions, targets, method_name):
    """Calculate and print metrics."""
    print(f"\n{method_name} Results:")
    print("--------------------------------------------------")
    target_names = ['Distal Error', 'Perp Error', 'Height Error', 'Missed Rate']
    metrics = {}
    for i, name in enumerate(target_names):
        mse = mean_squared_error(targets[:, i], predictions[:, i])
        r2 = r2_score(targets[:, i], predictions[:, i])
        metrics[name] = {'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2}
        print(f"{name}: MSE={mse:.4f}, RMSE={np.sqrt(mse):.4f}, R²={r2:.4f}")
    return metrics


def plot_comparison_results(
    distances, targets, predictions_dict, output_dir, class_names
):
    """Generate and save plots comparing model predictions."""
    target_names = ['Distal Error', 'Perpendicular Error', 'Height Error', 'Missed Rate']
    method_names = ['Neural Network', 'Linear/Quadratic']
    
    # Plot by class
    for class_name in ['Car', 'Pedestrian', 'Cyclist']:
        class_mask = np.array(class_names) == class_name
        
        if not np.any(class_mask):
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{class_name} - Error Prediction Comparison', fontsize=16)
        
        for i, target_name in enumerate(target_names):
            ax = axes[i // 2, i % 2]
            
            class_distances = distances[class_mask]
            class_targets = targets[class_mask]
            class_nn_pred = predictions_dict['Neural Network'][class_mask]
            class_linear_pred = predictions_dict['Linear/Quadratic'][class_mask]
            
            # Filter for valid predictions
            if i < 3:  # Error predictions
                valid_mask = class_targets[:, 3] == 0  # Only valid detections
                if np.any(valid_mask):
                    valid_distances = class_distances[valid_mask]
                    valid_targets = class_targets[valid_mask, i]
                    valid_nn_pred = class_nn_pred[valid_mask, i]
                    valid_linear_pred = class_linear_pred[valid_mask, i]
                else:
                    continue
            else:  # Missed rate prediction
                valid_distances = class_distances
                valid_targets = class_targets[:, i]
                valid_nn_pred = class_nn_pred[:, i]
                valid_linear_pred = class_linear_pred[:, i]
            
            # Plot actual vs predicted
            ax.scatter(valid_distances, valid_targets, alpha=0.6, s=20, 
                      label='Actual', color='blue')
            ax.scatter(valid_distances, valid_nn_pred, alpha=0.6, s=20, 
                      label='Neural Network', color='red', marker='s')
            ax.scatter(valid_distances, valid_linear_pred, alpha=0.6, s=20, 
                      label='Linear/Quadratic', color='green', marker='^')
            
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel(target_name)
            ax.set_title(f'{class_name} - {target_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{class_name.lower()}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Overall comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Overall Error Prediction Comparison', fontsize=16)
    
    for i, target_name in enumerate(target_names):
        ax = axes[i // 2, i % 2]
        
        # Filter for valid predictions
        if i < 3:  # Error predictions
            valid_mask = targets[:, 3] == 0  # Only valid detections
            if np.any(valid_mask):
                valid_targets = targets[valid_mask, i]
                valid_nn_pred = predictions_dict['Neural Network'][valid_mask, i]
                valid_linear_pred = predictions_dict['Linear/Quadratic'][valid_mask, i]
            else:
                continue
        else:  # Missed rate prediction
            valid_targets = targets[:, i]
            valid_nn_pred = predictions_dict['Neural Network'][:, i]
            valid_linear_pred = predictions_dict['Linear/Quadratic'][:, i]
        
        # Plot predictions vs actual
        ax.scatter(valid_targets, valid_nn_pred, alpha=0.6, s=20, 
                  label='Neural Network', color='red')
        ax.scatter(valid_targets, valid_linear_pred, alpha=0.6, s=20, 
                  label='Linear/Quadratic', color='green', marker='s')
        
        # Perfect prediction line
        min_val = min(valid_targets.min(), valid_nn_pred.min(), valid_linear_pred.min())
        max_val = max(valid_targets.max(), valid_nn_pred.max(), valid_linear_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax.set_xlabel(f'Actual {target_name}')
        ax.set_ylabel(f'Predicted {target_name}')
        ax.set_title(f'{target_name} - Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to: {output_dir}")


def main(args):
    """Main evaluation function."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    features, targets, distances, class_names = prepare_evaluation_data(
        args.data_root, args.ckpt, args.classes, args.iou_thresh
    )
    
    # Load NN model
    nn_model, scaler = load_trained_nn_model(
        os.path.join(args.model_dir, 'error_predictor.pth'),
        os.path.join(args.model_dir, 'scaler.pkl'),
        features.shape[1],
        args.hidden_dims
    )
    
    # Load linear models
    linear_models = load_linear_models(
        os.path.join(args.model_dir, 'linear_models.pkl')
    )

    # Evaluate models
    nn_predictions = evaluate_predictions_nn(nn_model, scaler, features)
    linear_predictions = evaluate_predictions_linear(linear_models, distances, targets)

    # Calculate metrics
    nn_metrics = calculate_metrics(nn_predictions, targets, "Neural Network")
    linear_metrics = calculate_metrics(linear_predictions, targets, "Linear/Quadratic")
    
    # Plot results
    plot_comparison_results(
        distances, 
        targets, 
        {
            'Neural Network': nn_predictions, 
            'Linear/Quadratic': linear_predictions
        },
        args.output_dir,
        class_names
    )

    # Save results
    results = {
        'nn_metrics': nn_metrics,
        'linear_metrics': linear_metrics,
        # ... (save other relevant data)
    }
    with open(os.path.join(args.output_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nEvaluation complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Error Predictor Models'
    )
    parser.add_argument(
        '--data_root', default='./data/kitti', help='Path to KITTI dataset'
    )
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='Path to PointPillars checkpoint')
    parser.add_argument(
        '--model_dir', default='error_predictor_results', 
        help='Directory where trained models are saved'
    )
    parser.add_argument(
        '--output_dir', default='error_predictor_evaluation', 
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--hidden_dims', nargs='+', type=int, default=[128, 64, 32], 
        help='NN hidden layer dimensions'
    )
    parser.add_argument(
        '--iou_thresh', type=float, default=0.3, help='IoU threshold for matching'
    )
    parser.add_argument(
        '--classes', nargs='+', default=['Car', 'Pedestrian', 'Cyclist'], 
        help='Classes to evaluate'
    )
    
    args = parser.parse_args()
    main(args) 

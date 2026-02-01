import argparse
import numpy as np
import os
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
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
    targets_list = []  # Absolute errors for regression comparison
    signed_targets_list = []  # Signed errors for distribution analysis
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
                for lidar_bbox, label, score in zip(lidar_bboxes, labels, scores):
                    format_result['name'].append(LABEL2CLASSES[label])
                    # Use lidar_bbox to match GT which is also in lidar coordinates
                    format_result['location'].append(lidar_bbox[:3])
                    format_result['dimensions'].append(lidar_bbox[3:6])
                    format_result['rotation_y'].append(lidar_bbox[6])
                    format_result['score'].append(score)
                
                # Convert to numpy arrays
                for key in format_result:
                    format_result[key] = np.array(format_result[key])
                
                for class_name in CLASSES:
                    gt_mask = gt_labels == CLASSES.index(class_name)
                    det_mask = format_result['name'] == class_name
                    
                    if not gt_mask.any():
                        continue
                    
                    # GT boxes are in lidar coordinates: [x, y, z, w, l, h, yaw]
                    # Both GT and detections should use the same format
                    gt_masked_cpu = gt_result[gt_mask].cpu().numpy()
                    # GT format: [x, y, z, w, l, h, yaw] - use as-is
                    gt_boxes = gt_masked_cpu  # Already in correct format
                    
                    if np.any(det_mask):
                        # Detection format from lidar_bbox: [x, y, z, w, l, h, yaw]
                        det_boxes = np.concatenate([
                            format_result['location'][det_mask],
                            format_result['dimensions'][det_mask],
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
                            det_box = det_boxes[gt_to_det[gt_idx]]
                            # Signed errors for distribution analysis
                            signed_errors = error_analyzer.compute_all_errors(gt_box, det_box, signed=True)
                            # Absolute errors for regression comparison
                            abs_errors = error_analyzer.compute_all_errors(gt_box, det_box, signed=False)
                            missed_rate = 0.0
                        else:
                            # Missed detection
                            det_box = None
                            signed_errors = {
                                'distal': np.nan, 'perp': np.nan, 'height': np.nan,
                                'yaw': np.nan, 'width': np.nan, 'length': np.nan, 
                                'box_height': np.nan
                            }
                            abs_errors = signed_errors.copy()
                            missed_rate = 1.0
                        
                        # Extract features
                        features = extract_features_from_detection(
                            gt_box, det_box, distance, class_name, CLASSES
                        )
                        
                        # Prepare targets: 7 absolute error types + missed_rate = 8 values
                        # These are used for regression model comparison
                        targets = [
                            abs_errors['distal'] if not np.isnan(abs_errors['distal']) else 0.0,
                            abs_errors['perp'] if not np.isnan(abs_errors['perp']) else 0.0,
                            abs_errors['height'] if not np.isnan(abs_errors['height']) else 0.0,
                            abs_errors['yaw'] if not np.isnan(abs_errors['yaw']) else 0.0,
                            abs_errors['width'] if not np.isnan(abs_errors['width']) else 0.0,
                            abs_errors['length'] if not np.isnan(abs_errors['length']) else 0.0,
                            abs_errors['box_height'] if not np.isnan(abs_errors['box_height']) else 0.0,
                            missed_rate
                        ]
                        
                        # Signed errors for distribution analysis (stored separately)
                        signed_targets = [
                            signed_errors['distal'] if not np.isnan(signed_errors['distal']) else np.nan,
                            signed_errors['perp'] if not np.isnan(signed_errors['perp']) else np.nan,
                            signed_errors['height'] if not np.isnan(signed_errors['height']) else np.nan,
                            signed_errors['yaw'] if not np.isnan(signed_errors['yaw']) else np.nan,
                            signed_errors['width'] if not np.isnan(signed_errors['width']) else np.nan,
                            signed_errors['length'] if not np.isnan(signed_errors['length']) else np.nan,
                            signed_errors['box_height'] if not np.isnan(signed_errors['box_height']) else np.nan,
                            missed_rate
                        ]
                        
                        features_list.append(features)
                        targets_list.append(targets)
                        signed_targets_list.append(signed_targets)
                        distances_list.append(distance)
                        class_names_list.append(class_name)
    
    return (np.array(features_list), np.array(targets_list), 
            np.array(signed_targets_list), np.array(distances_list), class_names_list)


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
    linear_predictions = np.zeros_like(targets)
    quadratic_predictions = np.zeros_like(targets)
    X = distances.reshape(-1, 1)
    
    # Target names: 7 error types + missed_rate
    target_names = ['distal', 'perp', 'height', 'yaw', 'width', 'length', 'box_height', 'missed_rate']
    
    for i, target_name in enumerate(target_names):
        linear_model = linear_models.get(f'{target_name}_linear')
        quad_model = linear_models.get(f'{target_name}_quadratic')

        if linear_model:
            linear_predictions[:, i] = linear_model.predict(X)
        if quad_model:
            quadratic_predictions[:, i] = quad_model.predict(X)

    return linear_predictions, quadratic_predictions


def calculate_metrics(predictions, targets, method_name):
    """Calculate and print metrics."""
    print(f"\n{method_name} Results:")
    print("--------------------------------------------------")
    target_names = ['Distal Error', 'Perp Error', 'Height Error', 'Yaw Error', 
                    'Width Error', 'Length Error', 'Box Height Error', 'Missed Rate']
    metrics = {}
    for i, name in enumerate(target_names):
        mse = mean_squared_error(targets[:, i], predictions[:, i])
        r2 = r2_score(targets[:, i], predictions[:, i])
        metrics[name] = {'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2}
        print(f"{name}: MSE={mse:.4f}, RMSE={np.sqrt(mse):.4f}, R²={r2:.4f}")
    return metrics


def fit_distributions(data):
    """Fit multiple distributions to data and return goodness-of-fit metrics."""
    results = {}
    
    # List of distributions to test
    distributions = {
        'Normal': stats.norm,
        'Laplace': stats.laplace,
        'Cauchy': stats.cauchy,
        'Student-t': stats.t,
        'Logistic': stats.logistic,
    }
    
    for name, dist in distributions.items():
        try:
            # Fit distribution
            params = dist.fit(data)
            
            # Calculate log-likelihood
            log_likelihood = np.sum(dist.logpdf(data, *params))
            
            # Calculate AIC and BIC
            k = len(params)  # number of parameters
            n = len(data)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(data, dist.cdf, args=params)
            
            results[name] = {
                'params': params,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'ks_stat': ks_stat,
                'ks_p': ks_p
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results


def analyze_error_distributions(distances, targets, output_dir, class_names, bin_size=1.0):
    """Analyze error distributions by distance bins using signed errors and multiple distribution fits."""
    target_names = ['Distal Error', 'Perpendicular Error', 'Height Error', 'Yaw Error',
                    'Width Error', 'Length Error', 'Box Height Error']
    class_names_arr = np.array(class_names)
    
    # Get valid detections only (not missed) - missed_rate is now at index 7
    valid_mask = targets[:, 7] == 0
    valid_distances = distances[valid_mask]
    valid_targets = targets[valid_mask, :7]  # Only error columns (0-6), not missed rate (7)
    valid_classes = class_names_arr[valid_mask]
    
    # Create distance bins
    min_dist = np.floor(valid_distances.min())
    max_dist = np.ceil(valid_distances.max())
    bins = np.arange(min_dist, max_dist + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    print("\n" + "="*80)
    print("ERROR DISTRIBUTION ANALYSIS BY DISTANCE (1m bins) - SIGNED ERRORS")
    print("="*80)
    print("\nTesting distributions: Normal, Laplace, Cauchy, Student-t, Logistic")
    print("Best fit determined by lowest AIC (Akaike Information Criterion)")
    
    # Store results for plotting
    distribution_results = {}
    
    for class_name in ['Car', 'Pedestrian', 'Cyclist', 'All']:
        if class_name == 'All':
            class_mask = np.ones(len(valid_classes), dtype=bool)
        else:
            class_mask = valid_classes == class_name
        
        if not np.any(class_mask):
            continue
            
        class_distances = valid_distances[class_mask]
        class_targets = valid_targets[class_mask]
        
        print(f"\n{'='*80}")
        print(f"{class_name} (n={np.sum(class_mask)})")
        print(f"{'='*80}")
        
        distribution_results[class_name] = {}
        
        for error_idx, error_name in enumerate(target_names):
            print(f"\n  {error_name}:")
            print(f"  {'Dist Bin':>10} {'Count':>6} {'Mean':>8} {'Std':>8} | {'Best Fit':>12} {'AIC':>10} {'KS p-val':>10}")
            print(f"  {'-'*80}")
            
            bin_stats = []
            
            for i in range(len(bins) - 1):
                bin_mask = (class_distances >= bins[i]) & (class_distances < bins[i+1])
                bin_errors = class_targets[bin_mask, error_idx]
                
                if len(bin_errors) >= 10:  # Need enough samples for distribution fitting
                    mean_err = np.mean(bin_errors)
                    std_err = np.std(bin_errors)
                    
                    # Fit multiple distributions
                    dist_fits = fit_distributions(bin_errors)
                    
                    # Find best fit by AIC
                    valid_fits = {k: v for k, v in dist_fits.items() if 'aic' in v}
                    if valid_fits:
                        best_fit = min(valid_fits.keys(), key=lambda k: valid_fits[k]['aic'])
                        best_aic = valid_fits[best_fit]['aic']
                        best_ks_p = valid_fits[best_fit]['ks_p']
                    else:
                        best_fit = "N/A"
                        best_aic = np.nan
                        best_ks_p = np.nan
                    
                    bin_stats.append({
                        'bin_center': bin_centers[i],
                        'bin_start': bins[i],
                        'bin_end': bins[i+1],
                        'count': len(bin_errors),
                        'mean': mean_err,
                        'std': std_err,
                        'distribution_fits': dist_fits,
                        'best_fit': best_fit,
                        'errors': bin_errors
                    })
                    
                    print(f"  {bins[i]:>4.0f}-{bins[i+1]:>4.0f}m {len(bin_errors):>6} "
                          f"{mean_err:>8.4f} {std_err:>8.4f} | {best_fit:>12} {best_aic:>10.1f} {best_ks_p:>10.4f}")
            
            distribution_results[class_name][error_name] = bin_stats
        
        # Summary of best distributions for this class
        print(f"\n  Summary for {class_name}:")
        for error_name in target_names:
            bin_stats = distribution_results[class_name].get(error_name, [])
            if bin_stats:
                best_fits = [b['best_fit'] for b in bin_stats if b['best_fit'] != 'N/A']
                if best_fits:
                    from collections import Counter
                    fit_counts = Counter(best_fits)
                    total = len(best_fits)
                    print(f"    {error_name}: ", end="")
                    print(", ".join([f"{k}: {v/total*100:.1f}%" for k, v in fit_counts.most_common()]))
    
    # Generate distribution plots
    _plot_error_distributions(distribution_results, output_dir)
    
    return distribution_results


def _plot_error_distributions(distribution_results, output_dir):
    """Plot error distributions with mean ± 1σ lines at 1m intervals for 0-50m range."""
    target_names = ['Distal Error', 'Perpendicular Error', 'Height Error', 'Yaw Error',
                    'Width Error', 'Length Error', 'Box Height Error']
    target_units = ['m', 'm', 'm', 'rad', 'm', 'm', 'm']
    
    # Colors for different distributions
    dist_colors = {
        'Normal': 'red',
        'Laplace': 'green',
        'Cauchy': 'orange',
        'Student-t': 'purple',
        'Logistic': 'brown'
    }
    
    dist_objects = {
        'Normal': stats.norm,
        'Laplace': stats.laplace,
        'Cauchy': stats.cauchy,
        'Student-t': stats.t,
        'Logistic': stats.logistic
    }
    
    for class_name, class_data in distribution_results.items():
        if class_name == 'All':
            continue  # Skip 'All' for individual class plots
            
        # Create figure with 2 rows x 4 columns (8 subplots, 7 error types + 1 for legend/info)
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle(f'{class_name} - Error Distributions (0-50m, 1m bins, Signed Errors)\nWith Mean (solid) and ±1σ (dashed) lines', fontsize=14)
        axes = axes.flatten()
        
        for error_idx, error_name in enumerate(target_names):
            ax = axes[error_idx]
            bin_stats = class_data.get(error_name, [])
            
            if not bin_stats:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(error_name)
                continue
            
            # Filter to 0-50m range
            filtered_bins = [b for b in bin_stats if b['bin_center'] <= 50]
            
            if not filtered_bins:
                ax.text(0.5, 0.5, 'No data in 0-50m range', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(error_name)
                continue
            
            # Collect all errors for histogram (0-50m)
            all_errors = []
            for b in filtered_bins:
                all_errors.extend(b['errors'].tolist())
            all_errors = np.array(all_errors)
            
            # Plot histogram of all errors in 0-50m range
            n, bins_hist, patches = ax.hist(all_errors, bins=50, density=True, 
                                             alpha=0.5, color='steelblue', 
                                             edgecolor='black', label='All (0-50m)')
            
            # Extract mean and std at 1m intervals
            distances = [b['bin_center'] for b in filtered_bins]
            means = [b['mean'] for b in filtered_bins]
            stds = [b['std'] for b in filtered_bins]
            
            # Fit overall distribution to all errors
            best_overall_fit = None
            best_aic = np.inf
            best_params = None
            for dist_name, dist_obj in dist_objects.items():
                try:
                    params = dist_obj.fit(all_errors)
                    log_likelihood = np.sum(dist_obj.logpdf(all_errors, *params))
                    k = len(params)
                    aic = 2 * k - 2 * log_likelihood
                    if aic < best_aic:
                        best_aic = aic
                        best_overall_fit = dist_name
                        best_params = params
                except:
                    pass
            
            # Plot best fit distribution
            if best_params is not None:
                x = np.linspace(all_errors.min() - 0.1, all_errors.max() + 0.1, 200)
                pdf = dist_objects[best_overall_fit].pdf(x, *best_params)
                ax.plot(x, pdf, color=dist_colors[best_overall_fit], linewidth=2.5,
                       label=f'Best: {best_overall_fit}')
            
            # Add zero line and overall stats in title
            overall_mean = np.mean(all_errors)
            overall_std = np.std(all_errors)
            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            
            unit = target_units[error_idx]
            ax.set_title(f'{error_name}\nμ={overall_mean:.4f}, σ={overall_std:.4f} {unit}', fontsize=11)
            ax.set_xlabel(f'{error_name} ({unit})')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # Use last subplot for mean/std vs distance plot
        ax = axes[7]
        ax.set_title('Mean and ±1σ vs Distance (all error types)', fontsize=11)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(target_names)))
        for error_idx, error_name in enumerate(target_names):
            bin_stats = class_data.get(error_name, [])
            if not bin_stats:
                continue
            
            filtered_bins = [b for b in bin_stats if b['bin_center'] <= 50]
            if not filtered_bins:
                continue
            
            distances = [b['bin_center'] for b in filtered_bins]
            means = [b['mean'] for b in filtered_bins]
            stds = [b['std'] for b in filtered_bins]
            
            ax.plot(distances, means, '-', color=colors[error_idx], linewidth=1.5, 
                   label=error_name.replace(' Error', ''))
            ax.fill_between(distances, 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           color=colors[error_idx], alpha=0.2)
        
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Error (m or rad)')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.legend(fontsize=7, loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{class_name.lower()}_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {class_name.lower()}_distributions.png")
    
    # Summary plot: Mean and Std vs Distance for all classes and error types
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('Mean and ±1σ Error vs Distance by Error Type (0-50m)', fontsize=14)
    axes = axes.flatten()
    
    class_colors = {'Car': 'blue', 'Pedestrian': 'green', 'Cyclist': 'orange'}
    
    for error_idx, error_name in enumerate(target_names):
        ax = axes[error_idx]
        
        for class_name in ['Car', 'Pedestrian', 'Cyclist']:
            if class_name not in distribution_results:
                continue
            bin_stats = distribution_results[class_name].get(error_name, [])
            if not bin_stats:
                continue
            
            # Filter to 0-50m
            filtered_bins = [b for b in bin_stats if b['bin_center'] <= 50]
            if not filtered_bins:
                continue
            
            distances = [b['bin_center'] for b in filtered_bins]
            means = [b['mean'] for b in filtered_bins]
            stds = [b['std'] for b in filtered_bins]
            
            color = class_colors[class_name]
            ax.plot(distances, means, '-', color=color, linewidth=1.5, label=f'{class_name} mean')
            ax.fill_between(distances, 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           color=color, alpha=0.2, label=f'{class_name} ±1σ')
        
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel(f'{target_units[error_idx]}')
        ax.set_title(error_name)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
    
    # Hide the 8th subplot
    axes[7].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved distribution_summary.png")
    
    # Create a detailed summary table
    _create_distribution_summary_table(distribution_results, output_dir)


def _format_distribution_equation(dist_name, params):
    """Format distribution equation with fitted parameters."""
    if dist_name == 'Normal':
        mu, sigma = params
        return f"f(x) = (1/(σ√(2π))) * exp(-(x-μ)²/(2σ²))  with μ={mu:.4f}, σ={sigma:.4f}"
    elif dist_name == 'Laplace':
        loc, scale = params
        return f"f(x) = (1/(2b)) * exp(-|x-μ|/b)  with μ={loc:.4f}, b={scale:.4f}"
    elif dist_name == 'Cauchy':
        loc, scale = params
        return f"f(x) = 1/(πγ(1+((x-x₀)/γ)²))  with x₀={loc:.4f}, γ={scale:.4f}"
    elif dist_name == 'Student-t':
        df, loc, scale = params
        return f"f(x) = Student-t(ν, μ, σ)  with ν={df:.2f}, μ={loc:.4f}, σ={scale:.4f}"
    elif dist_name == 'Logistic':
        loc, scale = params
        return f"f(x) = exp(-(x-μ)/s) / (s*(1+exp(-(x-μ)/s))²)  with μ={loc:.4f}, s={scale:.4f}"
    else:
        return f"Parameters: {params}"


def _create_distribution_summary_table(distribution_results, output_dir):
    """Create a text summary of distribution fits."""
    target_names = ['Distal Error', 'Perpendicular Error', 'Height Error', 'Yaw Error',
                    'Width Error', 'Length Error', 'Box Height Error']
    
    with open(os.path.join(output_dir, 'distribution_summary.txt'), 'w') as f:
        f.write("="*100 + "\n")
        f.write("ERROR DISTRIBUTION ANALYSIS SUMMARY (SIGNED ERRORS)\n")
        f.write("="*100 + "\n\n")
        
        for class_name in ['Car', 'Pedestrian', 'Cyclist', 'All']:
            if class_name not in distribution_results:
                continue
            
            f.write(f"\n{'='*80}\n")
            f.write(f"{class_name}\n")
            f.write(f"{'='*80}\n\n")
            
            for error_name in target_names:
                bin_stats = distribution_results[class_name].get(error_name, [])
                if not bin_stats:
                    continue
                
                # Count best fits
                from collections import Counter
                best_fits = [b['best_fit'] for b in bin_stats if b.get('best_fit') != 'N/A']
                fit_counts = Counter(best_fits)
                total = len(best_fits)
                
                f.write(f"\n{error_name}:\n")
                f.write("-" * 60 + "\n")
                
                if total > 0:
                    f.write("  Distribution fit summary (% of bins):\n")
                    for dist_name, count in fit_counts.most_common():
                        f.write(f"    {dist_name}: {count}/{total} ({count/total*100:.1f}%)\n")
                    
                    # Calculate mean bias (should be near 0 for unbiased)
                    all_means = [b['mean'] for b in bin_stats]
                    overall_mean = np.mean(all_means)
                    f.write(f"\n  Mean error across all bins: {overall_mean:.4f} m\n")
                    
                    # Calculate average std
                    all_stds = [b['std'] for b in bin_stats]
                    avg_std = np.mean(all_stds)
                    f.write(f"  Average std across all bins: {avg_std:.4f} m\n")
                    
                    # Check if std increases with distance
                    if len(bin_stats) > 5:
                        near_stds = [b['std'] for b in bin_stats[:len(bin_stats)//3]]
                        far_stds = [b['std'] for b in bin_stats[-len(bin_stats)//3:]]
                        near_avg = np.mean(near_stds)
                        far_avg = np.mean(far_stds)
                        f.write(f"  Std at near range (<{bin_stats[len(bin_stats)//3]['bin_center']:.0f}m): {near_avg:.4f} m\n")
                        f.write(f"  Std at far range (>{bin_stats[-len(bin_stats)//3]['bin_center']:.0f}m): {far_avg:.4f} m\n")
                        if far_avg > near_avg * 1.2:
                            f.write(f"  --> Heteroscedastic: std increases with distance by {(far_avg/near_avg - 1)*100:.1f}%\n")
                        else:
                            f.write(f"  --> Approximately homoscedastic\n")
                    
                    # Print fitted equations for each distance bin
                    f.write(f"\n  Fitted Distribution Equations by Distance Bin:\n")
                    f.write(f"  {'-'*90}\n")
                    for bin_data in bin_stats:
                        bin_start = bin_data.get('bin_start', bin_data['bin_center'] - 2.5)
                        bin_end = bin_data.get('bin_end', bin_data['bin_center'] + 2.5)
                        best_fit = bin_data.get('best_fit', 'N/A')
                        dist_fits = bin_data.get('distribution_fits', {})
                        
                        if best_fit != 'N/A' and best_fit in dist_fits:
                            params = dist_fits[best_fit].get('params', None)
                            if params:
                                equation = _format_distribution_equation(best_fit, params)
                                f.write(f"    {bin_start:>4.0f}-{bin_end:>4.0f}m: [{best_fit}]\n")
                                f.write(f"      {equation}\n")
                
                f.write("\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("DISTRIBUTION EQUATIONS REFERENCE:\n")
        f.write("="*100 + "\n")
        f.write("""
Normal Distribution:
  f(x) = (1/(σ√(2π))) * exp(-(x-μ)²/(2σ²))
  Parameters: μ (mean), σ (standard deviation)

Laplace Distribution:
  f(x) = (1/(2b)) * exp(-|x-μ|/b)
  Parameters: μ (location), b (scale)

Cauchy Distribution:
  f(x) = 1/(πγ(1+((x-x₀)/γ)²))
  Parameters: x₀ (location), γ (scale)

Student-t Distribution:
  f(x) = Γ((ν+1)/2) / (√(νπ)Γ(ν/2)) * (1+(x-μ)²/(νσ²))^(-(ν+1)/2)
  Parameters: ν (degrees of freedom), μ (location), σ (scale)
  Note: As ν→∞, Student-t approaches Normal

Logistic Distribution:
  f(x) = exp(-(x-μ)/s) / (s*(1+exp(-(x-μ)/s))²)
  Parameters: μ (location), s (scale)

""")
        
        f.write("\n" + "="*100 + "\n")
        f.write("INTERPRETATION GUIDE:\n")
        f.write("="*100 + "\n")
        f.write("""
- Normal: Errors are Gaussian distributed (most common assumption)
- Laplace: Heavier tails than Normal, often fits when there are outliers
- Cauchy: Very heavy tails, robust to extreme outliers
- Student-t: Generalizes Normal with adjustable tail heaviness
- Logistic: Similar to Normal but slightly heavier tails

If Laplace or Student-t fits best, consider:
- Using L1 loss (MAE) instead of L2 loss (MSE) for training
- The error model should account for outliers

If errors are heteroscedastic (variance increases with distance):
- Consider distance-dependent error models
- Weight predictions by distance for uncertainty estimation
""")
    
    print("Saved distribution_summary.txt")


def save_distributions_csv(distribution_results, linear_models, output_dir):
    """Export distribution fits to CSV format compatible with sensor model."""
    # Map from our names to CSV names
    error_type_map = {
        'Distal Error': 'distal',
        'Perpendicular Error': 'perpendicular', 
        'Height Error': 'height',
        'Yaw Error': 'yaw',
        'Width Error': 'width',
        'Length Error': 'length',
        'Box Height Error': 'box_height'
    }
    
    dist_name_map = {
        'Normal': 'normal',
        'Laplace': 'laplace',
        'Logistic': 'logistic',
        'Student-t': 'student_t',
        'Cauchy': 'cauchy'
    }
    
    with open(os.path.join(output_dir, 'pointpillars_distributions.csv'), 'w') as f:
        # Write header
        f.write("# PointPillars KITTI - Best-Fit Error Distributions\n")
        f.write("# These are used for SAMPLING actual errors (not prediction)\n")
        f.write("# Based on evaluation with 'All' vehicle category\n")
        f.write("#\n")
        f.write("# Format: error_type,dist_min,dist_max,distribution,param1,param2,param3\n")
        f.write("#   distribution types: normal, laplace, logistic, student_t, cauchy\n")
        f.write("#   normal:    param1=mu, param2=sigma\n")
        f.write("#   laplace:   param1=mu, param2=b (scale)\n")
        f.write("#   logistic:  param1=mu, param2=s (scale)\n")
        f.write("#   student_t: param1=nu (df), param2=mu, param3=sigma\n")
        f.write("#   cauchy:    param1=x0 (location), param2=gamma (scale)\n")
        f.write("\n")
        f.write("error_type,dist_min,dist_max,distribution,param1,param2,param3\n")
        
        # Use 'All' class for the combined data
        if 'All' not in distribution_results:
            print("Warning: 'All' class not found in distribution results")
            return
        
        all_data = distribution_results['All']
        
        for error_name, csv_name in error_type_map.items():
            f.write(f"# ==============================================================================\n")
            f.write(f"# {error_name} bins\n")
            f.write(f"# ==============================================================================\n")
            
            bin_stats = all_data.get(error_name, [])
            
            if not bin_stats:
                f.write(f"# No data available for {error_name}\n")
                continue
            
            # Group bins into ranges (5m bins for CSV to reduce rows)
            # First, collect all 1m bin data
            bin_data_by_range = {}
            for b in bin_stats:
                # Group into 5m ranges
                range_start = int(b['bin_start'] // 5) * 5
                range_end = range_start + 5
                key = (range_start, range_end)
                
                if key not in bin_data_by_range:
                    bin_data_by_range[key] = []
                bin_data_by_range[key].append(b)
            
            # For each 5m range, use the most common best fit and average params
            for (range_start, range_end) in sorted(bin_data_by_range.keys()):
                bins_in_range = bin_data_by_range[(range_start, range_end)]
                
                # Count best fits in this range
                from collections import Counter
                best_fits = [b['best_fit'] for b in bins_in_range if b.get('best_fit') != 'N/A']
                if not best_fits:
                    continue
                
                # Use most common distribution
                fit_counts = Counter(best_fits)
                best_dist = fit_counts.most_common(1)[0][0]
                csv_dist = dist_name_map.get(best_dist, 'normal')
                
                # Collect all errors in this range and fit the distribution
                all_errors = []
                for b in bins_in_range:
                    all_errors.extend(b['errors'].tolist())
                all_errors = np.array(all_errors)
                
                # Fit the selected distribution to get params
                try:
                    if best_dist == 'Normal':
                        params = stats.norm.fit(all_errors)
                        f.write(f"{csv_name},{range_start},{range_end},{csv_dist},{params[0]:.4f},{params[1]:.4f},\n")
                    elif best_dist == 'Laplace':
                        params = stats.laplace.fit(all_errors)
                        f.write(f"{csv_name},{range_start},{range_end},{csv_dist},{params[0]:.4f},{params[1]:.4f},\n")
                    elif best_dist == 'Logistic':
                        params = stats.logistic.fit(all_errors)
                        f.write(f"{csv_name},{range_start},{range_end},{csv_dist},{params[0]:.4f},{params[1]:.4f},\n")
                    elif best_dist == 'Student-t':
                        params = stats.t.fit(all_errors)
                        f.write(f"{csv_name},{range_start},{range_end},{csv_dist},{params[0]:.2f},{params[1]:.4f},{params[2]:.4f}\n")
                    elif best_dist == 'Cauchy':
                        params = stats.cauchy.fit(all_errors)
                        f.write(f"{csv_name},{range_start},{range_end},{csv_dist},{params[0]:.4f},{params[1]:.4f},\n")
                except Exception as e:
                    # Fallback to normal
                    mu, sigma = np.mean(all_errors), np.std(all_errors)
                    f.write(f"{csv_name},{range_start},{range_end},normal,{mu:.4f},{sigma:.4f},\n")
        
        # Add missed detection section
        f.write(f"# ==============================================================================\n")
        f.write(f"# Missed Detection bins (probability of NOT detecting at each distance)\n")
        f.write(f"# This represents detection failures, sampled as Bernoulli with these probabilities\n")
        f.write(f"# ==============================================================================\n")
        
        # Use linear model to compute miss rates at different distances
        miss_model = linear_models.get('missed_rate_linear')
        if miss_model:
            coef = miss_model.coef_[0]
            intercept = miss_model.intercept_
            
            # Generate miss rates for different distance ranges
            ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]
            for d_min, d_max in ranges:
                d_mid = (d_min + d_max) / 2
                miss_prob = coef * d_mid + intercept
                miss_prob = np.clip(miss_prob, 0, 1)  # Ensure valid probability
                f.write(f"miss_detection,{d_min},{d_max},normal,{miss_prob:.3f},0.05,\n")
        else:
            # Fallback values
            f.write("miss_detection,0,10,normal,0.50,0.05,\n")
            f.write("miss_detection,10,20,normal,0.50,0.05,\n")
            f.write("miss_detection,20,30,normal,0.50,0.05,\n")
            f.write("miss_detection,30,40,normal,0.50,0.05,\n")
            f.write("miss_detection,40,50,normal,0.50,0.05,\n")
            f.write("miss_detection,50,100,normal,0.50,0.05,\n")
    
    print("Saved pointpillars_distributions.csv")


def save_regression_models_summary(linear_models, output_dir):
    """Save all linear/quadratic regression model parameters to the summary file."""
    target_names = ['distal', 'perp', 'height', 'yaw', 'width', 'length', 'box_height', 'missed_rate']
    target_display = {
        'distal': 'Radial Error (range, towards/away from sensor)',
        'perp': 'Lateral Error (cross-range, left/right in ground plane)', 
        'height': 'Vertical Error (elevation, up/down)',
        'yaw': 'Yaw Error (heading angle, radians)',
        'width': 'Width Error (box dimension)',
        'length': 'Length Error (box dimension)',
        'box_height': 'Box Height Error (box dimension)',
        'missed_rate': 'Missed Detection Rate'
    }
    
    with open(os.path.join(output_dir, 'regression_models.txt'), 'w') as f:
        f.write("="*100 + "\n")
        f.write("REGRESSION MODEL PARAMETERS\n")
        f.write("="*100 + "\n\n")
        f.write("All models predict error as a function of distance (d) in meters.\n\n")
        
        for target_name in target_names:
            display_name = target_display.get(target_name, target_name)
            f.write(f"\n{'='*80}\n")
            f.write(f"{display_name}\n")
            f.write(f"{'='*80}\n\n")
            
            # Linear model
            linear_model = linear_models.get(f'{target_name}_linear')
            if linear_model:
                coef = linear_model.coef_[0]
                intercept = linear_model.intercept_
                f.write(f"  LINEAR MODEL:\n")
                f.write(f"    error = {coef:.6f} * d + {intercept:.6f}\n")
                f.write(f"    Coefficient: {coef:.6f}\n")
                f.write(f"    Intercept: {intercept:.6f}\n\n")
            
            # Quadratic model
            quad_model = linear_models.get(f'{target_name}_quadratic')
            if quad_model:
                # Extract coefficients from pipeline
                poly_features = quad_model.named_steps['poly']
                linear_reg = quad_model.named_steps['linear']
                # Coefficients are [1, d, d^2] for degree=2
                coefs = linear_reg.coef_
                intercept = linear_reg.intercept_
                f.write(f"  QUADRATIC MODEL:\n")
                f.write(f"    error = {coefs[2]:.10f} * d² + {coefs[1]:.10f} * d + {intercept:.10f}\n")
                f.write(f"    Coefficient (d²): {coefs[2]:.10e}\n")
                f.write(f"    Coefficient (d): {coefs[1]:.10e}\n")
                f.write(f"    Intercept: {intercept:.10f}\n\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("USAGE NOTES:\n")
        f.write("="*100 + "\n")
        f.write("""
To predict error for a given distance d:
  Linear:    error(d) = coef * d + intercept
  Quadratic: error(d) = coef_d2 * d^2 + coef_d * d + intercept

Units:
  - Distance (d): meters
  - Position errors (distal, perp, height): meters
  - Yaw error: radians
  - Dimension errors (width, length, box_height): meters
  - Missed rate: probability (0-1)

These are ABSOLUTE error magnitude predictions (always positive).
For signed error distributions (bias analysis), use distribution_summary.txt.
""")
    
    print("Saved regression_models.txt")


def plot_comparison_results(
    distances, targets, predictions_dict, output_dir, class_names, linear_models
):
    """Generate and save plots comparing model predictions."""
    target_names = ['Distal Error', 'Perpendicular Error', 'Height Error', 'Yaw Error',
                    'Width Error', 'Length Error', 'Box Height Error', 'Missed Rate']
    target_keys = ['distal', 'perp', 'height', 'yaw', 'width', 'length', 'box_height', 'missed_rate']
    
    # Debug: print class distribution
    class_names_arr = np.array(class_names)
    print(f"Class distribution: Car={np.sum(class_names_arr == 'Car')}, "
          f"Pedestrian={np.sum(class_names_arr == 'Pedestrian')}, "
          f"Cyclist={np.sum(class_names_arr == 'Cyclist')}")
    
    # Plot by class
    for class_name in ['Car', 'Pedestrian', 'Cyclist']:
        class_mask = class_names_arr == class_name
        
        if not np.any(class_mask):
            print(f"No data for class {class_name}, skipping...")
            continue
        
        # Create 2x4 grid for 8 error types
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.flatten()
        fig.suptitle(f'{class_name} - Error Prediction Comparison', fontsize=16)
        
        for i, target_name in enumerate(target_names):
            ax = axes[i]
            
            class_distances = distances[class_mask]
            class_targets = targets[class_mask]
            
            # Filter for valid predictions
            if i < 7:  # Error predictions (all 7 error types)
                valid_mask = class_targets[:, 7] == 0  # Only valid detections (not missed)
                if np.any(valid_mask):
                    valid_distances = class_distances[valid_mask]
                    valid_targets = class_targets[valid_mask, i]
                else:
                    # No valid detections, show message on plot
                    ax.text(0.5, 0.5, 'No valid detections', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{target_name}')
                    continue
            else:  # Missed rate prediction (index 7)
                valid_distances = class_distances
                valid_targets = class_targets[:, i]
            
            # Get NN predictions for this class/target
            class_nn_pred = predictions_dict['Neural Network'][class_mask]
            if i < 7:
                valid_nn_pred = class_nn_pred[valid_mask, i]
            else:
                valid_nn_pred = class_nn_pred[:, i]
            
            # Plot actual data points
            ax.scatter(valid_distances, valid_targets, alpha=0.3, s=10, 
                      label='Actual', color='blue')
            
            # Plot NN predictions as scatter
            ax.scatter(valid_distances, valid_nn_pred, alpha=0.3, s=10, 
                      label='NN Predicted', color='orange', marker='x')
            
            # Generate smooth distance range for regression lines
            dist_min, dist_max = valid_distances.min(), valid_distances.max()
            dist_range = np.linspace(dist_min, dist_max, 100).reshape(-1, 1)
            
            # Plot linear regression line
            linear_model = linear_models.get(f'{target_keys[i]}_linear')
            if linear_model:
                linear_line = linear_model.predict(dist_range)
                ax.plot(dist_range, linear_line, 'g-', linewidth=2, 
                       label='Linear Fit')
            
            # Plot quadratic regression line
            quad_model = linear_models.get(f'{target_keys[i]}_quadratic')
            if quad_model:
                quad_line = quad_model.predict(dist_range)
                ax.plot(dist_range, quad_line, 'r-', linewidth=2, 
                       label='Quadratic Fit')
            
            # Calculate and plot mean, median and ±1σ lines at 1m intervals
            bin_edges = np.arange(0, valid_distances.max() + 1, 1)
            bin_centers = []
            bin_means = []
            bin_medians = []
            bin_stds = []
            
            for j in range(len(bin_edges) - 1):
                bin_mask = (valid_distances >= bin_edges[j]) & (valid_distances < bin_edges[j+1])
                if np.sum(bin_mask) >= 3:  # Need at least 3 samples
                    bin_centers.append((bin_edges[j] + bin_edges[j+1]) / 2)
                    bin_means.append(np.mean(valid_targets[bin_mask]))
                    bin_medians.append(np.median(valid_targets[bin_mask]))
                    bin_stds.append(np.std(valid_targets[bin_mask]))
            
            if len(bin_centers) > 0:
                bin_centers = np.array(bin_centers)
                bin_means = np.array(bin_means)
                bin_medians = np.array(bin_medians)
                bin_stds = np.array(bin_stds)
                
                # Plot median line (should be closer to regression for skewed data)
                ax.plot(bin_centers, bin_medians, 'k-', linewidth=2.5, label='Median (1m bins)')
                # Plot mean line
                ax.plot(bin_centers, bin_means, 'k--', linewidth=1.5, label='Mean (1m bins)')
                # Plot ±1σ bands around median
                ax.fill_between(bin_centers, bin_medians - bin_stds, bin_medians + bin_stds,
                               color='gray', alpha=0.3, label='±1σ')
            
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel(target_name)
            ax.set_title(f'{class_name} - {target_name}')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{class_name.lower()}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {class_name.lower()}_comparison.png")
    
    # Overall comparison plot - Error vs Distance with regression lines
    # Create 2x4 grid for 8 error types
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    fig.suptitle('Overall Error vs Distance (All Classes)', fontsize=16)
    
    for i, target_name in enumerate(target_names):
        ax = axes[i]
        
        # Filter for valid predictions
        if i < 7:  # Error predictions (7 error types)
            valid_mask = targets[:, 7] == 0  # Only valid detections
            if np.any(valid_mask):
                valid_distances = distances[valid_mask]
                valid_targets = targets[valid_mask, i]
            else:
                ax.text(0.5, 0.5, 'No valid detections', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{target_name}')
                continue
        else:  # Missed rate prediction (index 7)
            valid_distances = distances
            valid_targets = targets[:, i]
        
        # Get NN predictions for this target
        if i < 7:
            valid_nn_pred = predictions_dict['Neural Network'][valid_mask, i]
        else:
            valid_nn_pred = predictions_dict['Neural Network'][:, i]
        
        # Plot actual data points
        ax.scatter(valid_distances, valid_targets, alpha=0.2, s=8, 
                  label='Actual', color='blue')
        
        # Plot NN predictions as scatter
        ax.scatter(valid_distances, valid_nn_pred, alpha=0.2, s=8, 
                  label='NN Predicted', color='orange', marker='x')
        
        # Generate smooth distance range for regression lines
        dist_min, dist_max = valid_distances.min(), valid_distances.max()
        dist_range = np.linspace(dist_min, dist_max, 100).reshape(-1, 1)
        
        # Plot linear regression line
        linear_model = linear_models.get(f'{target_keys[i]}_linear')
        if linear_model:
            linear_line = linear_model.predict(dist_range)
            ax.plot(dist_range, linear_line, 'g-', linewidth=2.5, 
                   label='Linear Fit')
        
        # Plot quadratic regression line
        quad_model = linear_models.get(f'{target_keys[i]}_quadratic')
        if quad_model:
            quad_line = quad_model.predict(dist_range)
            ax.plot(dist_range, quad_line, 'r-', linewidth=2.5, 
                   label='Quadratic Fit')
        
        # Calculate and plot mean, median and ±1σ lines at 1m intervals
        bin_edges = np.arange(0, valid_distances.max() + 1, 1)
        bin_centers = []
        bin_means = []
        bin_medians = []
        bin_stds = []
        
        for j in range(len(bin_edges) - 1):
            bin_mask = (valid_distances >= bin_edges[j]) & (valid_distances < bin_edges[j+1])
            if np.sum(bin_mask) >= 3:  # Need at least 3 samples
                bin_centers.append((bin_edges[j] + bin_edges[j+1]) / 2)
                bin_means.append(np.mean(valid_targets[bin_mask]))
                bin_medians.append(np.median(valid_targets[bin_mask]))
                bin_stds.append(np.std(valid_targets[bin_mask]))
        
        if len(bin_centers) > 0:
            bin_centers_arr = np.array(bin_centers)
            bin_means_arr = np.array(bin_means)
            bin_medians_arr = np.array(bin_medians)
            bin_stds_arr = np.array(bin_stds)
            
            # Plot median line (should be closer to regression for skewed data)
            ax.plot(bin_centers_arr, bin_medians_arr, 'k-', linewidth=2.5, label='Median (1m bins)')
            # Plot mean line
            ax.plot(bin_centers_arr, bin_means_arr, 'k--', linewidth=1.5, label='Mean (1m bins)')
            # Plot ±1σ bands around median
            ax.fill_between(bin_centers_arr, bin_medians_arr - bin_stds_arr, bin_medians_arr + bin_stds_arr,
                           color='gray', alpha=0.3, label='±1σ')
        
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel(target_name)
        ax.set_title(f'{target_name} vs Distance')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overall_comparison.png")
    print(f"All plots saved to: {output_dir}")


def main(args):
    """Main evaluation function."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # targets = absolute errors for regression comparison
    # signed_targets = signed errors for distribution analysis
    features, targets, signed_targets, distances, class_names = prepare_evaluation_data(
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

    # Evaluate models using absolute error targets
    nn_predictions = evaluate_predictions_nn(nn_model, scaler, features)
    linear_predictions, quadratic_predictions = evaluate_predictions_linear(linear_models, distances, targets)

    # Calculate metrics using absolute errors
    nn_metrics = calculate_metrics(nn_predictions, targets, "Neural Network")
    linear_metrics = calculate_metrics(linear_predictions, targets, "Linear")
    quadratic_metrics = calculate_metrics(quadratic_predictions, targets, "Quadratic")
    
    # Plot results using absolute errors
    plot_comparison_results(
        distances, 
        targets, 
        {
            'Neural Network': nn_predictions, 
            'Linear': linear_predictions,
            'Quadratic': quadratic_predictions
        },
        args.output_dir,
        class_names,
        linear_models
    )
    
    # Analyze error distributions using SIGNED errors for histograms
    distribution_results = analyze_error_distributions(
        distances, signed_targets, args.output_dir, class_names, bin_size=1.0
    )
    
    # Save regression model parameters to file
    save_regression_models_summary(linear_models, args.output_dir)
    
    # Save distributions to CSV format
    save_distributions_csv(distribution_results, linear_models, args.output_dir)

    # Save results
    results = {
        'nn_metrics': nn_metrics,
        'linear_metrics': linear_metrics,
        'quadratic_metrics': quadratic_metrics,
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

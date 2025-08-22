import argparse
import numpy as np
import os
import torch
from tqdm import tqdm
from collections import defaultdict

from pointpillars.utils import keep_bbox_from_image_range, \
    keep_bbox_from_lidar_range, write_pickle, write_label, iou3d_camera
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars


class ErrorAnalyzer:
    """Analyzes empirical LIDAR detection errors and missed detection rates."""
    def __init__(self):
        self.distance_bins = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        self.distance_bin_centers = (self.distance_bins[:-1] + 
                                    self.distance_bins[1:]) / 2

    def calculate_distance(self, location):
        return np.sqrt(location[:, 0] ** 2 + location[:, 2] ** 2)

    def match_detections(self, gt_boxes, det_boxes, iou_thresh=0.5):
        """Match detections to ground truth using 3D IoU."""
        if len(gt_boxes) == 0 or len(det_boxes) == 0:
            return {}, set(range(len(gt_boxes))), set()
        # Convert to tensors without extra dimension and move to CUDA
        gt_tensor = torch.from_numpy(gt_boxes).float().cuda()
        det_tensor = torch.from_numpy(det_boxes).float().cuda()
        ious = iou3d_camera(gt_tensor, det_tensor).cpu().numpy()
        gt_to_det = {}
        used_dets = set()
        for gt_idx in range(len(gt_boxes)):
            det_idx = np.argmax(ious[gt_idx])
            if ious[gt_idx, det_idx] >= iou_thresh and det_idx not in used_dets:
                gt_to_det[gt_idx] = det_idx
                used_dets.add(det_idx)
        unmatched_gt = set(range(len(gt_boxes))) - set(gt_to_det.keys())
        unmatched_det = set(range(len(det_boxes))) - used_dets
        return gt_to_det, unmatched_gt, unmatched_det

    def compute_errors(self, gt_box, det_box):
        """Compute distal, perpendicular, and height errors between GT and detection."""
        gt_center = gt_box[:3]
        det_center = det_box[:3]
        # Vector from sensor to GT
        sensor_to_gt = gt_center.copy()
        sensor_to_gt[1] = 0  # project to ground plane
        norm = np.linalg.norm(sensor_to_gt)
        if norm == 0:
            direction = np.array([1, 0, 0])
        else:
            direction = sensor_to_gt / norm
        # Vector from GT to detection
        delta = det_center - gt_center
        # Distal error (projection along direction) - use absolute value
        distal_error = abs(np.dot(delta[[0, 2]], direction[[0, 2]]))
        # Perpendicular error (in ground plane, orthogonal to direction) - use absolute value
        perp_vec = np.array([-direction[2], 0, direction[0]])
        perp_error = abs(np.dot(delta[[0, 2]], perp_vec[[0, 2]]))
        # Height error - use absolute value
        height_error = abs(delta[1])
        return distal_error, perp_error, height_error

    def analyze_detection_errors(self, gt_results, det_results, CLASSES, iou_thresh):
        error_analysis = {
            'per_class': defaultdict(lambda: {
                'distances': [],
                'distal_errors': [],
                'perp_errors': [],
                'height_errors': [],
                'missed_detections': 0,
                'total_gt': 0,
                'total_detections': 0
            })
        }
        ids = list(sorted(gt_results.keys()))
        for id in ids:
            gt_result = gt_results[id]['annos']
            det_result = det_results[id]
            for class_name in CLASSES:
                gt_mask = gt_result['name'] == class_name
                det_mask = det_result['name'] == class_name
                # Format boxes as [x, y, z, w, l, h, rotation] for iou3d_camera
                gt_boxes = np.concatenate([
                    gt_result['location'][gt_mask],
                    gt_result['dimensions'][gt_mask][:, [1, 0, 2]],  # reorder: l,w,h -> w,l,h
                    gt_result['rotation_y'][gt_mask][:, None]
                ], axis=1)
                det_boxes = np.concatenate([
                    det_result['location'][det_mask],
                    det_result['dimensions'][det_mask][:, [1, 0, 2]],  # reorder: l,w,h -> w,l,h
                    det_result['rotation_y'][det_mask][:, None]
                ], axis=1)
                gt_distances = self.calculate_distance(gt_result['location'][gt_mask])
                error_analysis['per_class'][class_name]['total_gt'] += len(gt_boxes)
                error_analysis['per_class'][class_name]['total_detections'] += len(det_boxes)
                if len(gt_boxes) == 0:
                    continue
                gt_to_det, unmatched_gt, _ = self.match_detections(
                    gt_boxes, det_boxes, iou_thresh=iou_thresh
                )
                for gt_idx, gt_box in enumerate(gt_boxes):
                    error_analysis['per_class'][class_name]['distances'].append(gt_distances[gt_idx])
                    if gt_idx in gt_to_det:
                        det_box = det_boxes[gt_to_det[gt_idx]]
                        distal, perp, height = self.compute_errors(gt_box, det_box)
                        error_analysis['per_class'][class_name]['distal_errors'].append(distal)
                        error_analysis['per_class'][class_name]['perp_errors'].append(perp)
                        error_analysis['per_class'][class_name]['height_errors'].append(height)
                    else:
                        error_analysis['per_class'][class_name]['distal_errors'].append(np.nan)
                        error_analysis['per_class'][class_name]['perp_errors'].append(np.nan)
                        error_analysis['per_class'][class_name]['height_errors'].append(np.nan)
                        error_analysis['per_class'][class_name]['missed_detections'] += 1
        return error_analysis

    def calculate_distance_binned_errors(self, error_analysis):
        binned_errors = {'per_class': {}}
        for class_name in error_analysis['per_class']:
            distances = np.array(error_analysis['per_class'][class_name]['distances'])
            distal = np.array(error_analysis['per_class'][class_name]['distal_errors'])
            perp = np.array(error_analysis['per_class'][class_name]['perp_errors'])
            height = np.array(error_analysis['per_class'][class_name]['height_errors'])
            if len(distances) == 0:
                continue
            binned_errors['per_class'][class_name] = {
                'distance_bins': self.distance_bin_centers,
                'distal_mean': [], 'distal_std': [],
                'perp_mean': [], 'perp_std': [],
                'height_mean': [], 'height_std': [],
                'counts': [], 'missed': []
            }
            for i in range(len(self.distance_bins) - 1):
                mask = (distances >= self.distance_bins[i]) & (distances < self.distance_bins[i + 1])
                valid = mask & ~np.isnan(distal)
                missed = np.sum(mask & np.isnan(distal))
                binned_errors['per_class'][class_name]['counts'].append(np.sum(mask))
                binned_errors['per_class'][class_name]['missed'].append(missed)
                if np.any(valid):
                    binned_errors['per_class'][class_name]['distal_mean'].append(np.nanmean(distal[valid]))
                    binned_errors['per_class'][class_name]['distal_std'].append(np.nanstd(distal[valid]))
                    binned_errors['per_class'][class_name]['perp_mean'].append(np.nanmean(perp[valid]))
                    binned_errors['per_class'][class_name]['perp_std'].append(np.nanstd(perp[valid]))
                    binned_errors['per_class'][class_name]['height_mean'].append(np.nanmean(height[valid]))
                    binned_errors['per_class'][class_name]['height_std'].append(np.nanstd(height[valid]))
                else:
                    binned_errors['per_class'][class_name]['distal_mean'].append(np.nan)
                    binned_errors['per_class'][class_name]['distal_std'].append(np.nan)
                    binned_errors['per_class'][class_name]['perp_mean'].append(np.nan)
                    binned_errors['per_class'][class_name]['perp_std'].append(np.nan)
                    binned_errors['per_class'][class_name]['height_mean'].append(np.nan)
                    binned_errors['per_class'][class_name]['height_std'].append(np.nan)
        return binned_errors

    def print_error_summary(self, error_analysis, binned_errors):
        print("\n" + "="*80)
        print("EMPIRICAL LIDAR ERROR ANALYSIS SUMMARY")
        print("="*80)
        print(f"\nPER-CLASS STATISTICS:")
        print("-" * 60)
        print(f"{'Class':<12} {'GT':<6} {'Det':<6} {'Missed':<8} {'Miss Rate':<10}")
        print("-" * 60)
        for class_name, stats in error_analysis['per_class'].items():
            if stats['total_gt'] > 0:
                missed_rate = stats['missed_detections'] / stats['total_gt'] * 100
                print(f"{class_name:<12} {stats['total_gt']:<6} {stats['total_detections']:<6} "
                      f"{stats['missed_detections']:<8} {missed_rate:<10.2f}%")
        print(f"\nEMPIRICAL ERROR ANALYSIS BY DISTANCE:")
        print("-" * 80)
        for class_name in ['Car', 'Pedestrian', 'Cyclist']:
            if class_name in binned_errors['per_class']:
                print(f"\n{class_name} Errors by Distance:")
                print(f"{'Distance':<10} {'Distal Mean':<12} {'Distal Std':<12} {'Perp Mean':<12} {'Perp Std':<12} {'Height Mean':<12} {'Height Std':<12} {'Missed':<8} {'Count':<8}")
                print("-" * 100)
                for i, distance in enumerate(binned_errors['per_class'][class_name]['distance_bins']):
                    count = binned_errors['per_class'][class_name]['counts'][i]
                    missed = binned_errors['per_class'][class_name]['missed'][i]
                    if count > 0:
                        distal_mean = binned_errors['per_class'][class_name]['distal_mean'][i]
                        distal_std = binned_errors['per_class'][class_name]['distal_std'][i]
                        perp_mean = binned_errors['per_class'][class_name]['perp_mean'][i]
                        perp_std = binned_errors['per_class'][class_name]['perp_std'][i]
                        height_mean = binned_errors['per_class'][class_name]['height_mean'][i]
                        height_std = binned_errors['per_class'][class_name]['height_std'][i]
                        print(f"{distance:<10.1f} {distal_mean:<12.4f} {distal_std:<12.4f} {perp_mean:<12.4f} {perp_std:<12.4f} {height_mean:<12.4f} {height_std:<12.4f} {missed:<8} {count:<8}")

    def save_error_analysis(self, error_analysis, binned_errors, saved_path):
        results = {
            'error_analysis': dict(error_analysis),
            'binned_errors': binned_errors
        }
        for category in results['error_analysis']:
            if category == 'per_class':
                results['error_analysis'][category] = dict(results['error_analysis'][category])
        write_pickle(results, os.path.join(saved_path, 'error_analysis.pkl'))
        with open(os.path.join(saved_path, 'error_summary.txt'), 'w') as f:
            f.write("EMPIRICAL LIDAR ERROR ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n")
            for class_name, stats in error_analysis['per_class'].items():
                if stats['total_gt'] > 0:
                    missed_rate = stats['missed_detections'] / stats['total_gt'] * 100
                    f.write(f"\n{class_name}: Miss Rate = {missed_rate:.2f}% "
                           f"(GT: {stats['total_gt']}, Det: {stats['total_detections']})\n")

def main(args):
    val_dataset = Kitti(data_root=args.data_root, split='val')
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)
    CLASSES = Kitti.CLASSES
    LABEL2CLASSES = {v: k for k, v in CLASSES.items()}
    if not args.no_cuda:
        model = PointPillars(nclasses=args.nclasses).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=args.nclasses)
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)
    saved_submit_path = os.path.join(saved_path, 'submit')
    os.makedirs(saved_submit_path, exist_ok=True)
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)
    error_analyzer = ErrorAnalyzer()
    model.eval()
    format_results = {}
    print('Running inference and collecting results...')
    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(val_dataloader)):
            if not args.no_cuda:
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batch_results = model(batched_pts=batched_pts, 
                                  mode='val',
                                  batched_gt_bboxes=batched_gt_bboxes, 
                                  batched_gt_labels=batched_labels)
            for j, result in enumerate(batch_results):
                format_result = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }
                calib_info = data_dict['batched_calib_info'][j]
                tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
                r0_rect = calib_info['R0_rect'].astype(np.float32)
                P2 = calib_info['P2'].astype(np.float32)
                image_shape = data_dict['batched_img_info'][j]['image_shape']
                idx = data_dict['batched_img_info'][j]['image_idx']
                result_filter = keep_bbox_from_image_range(
                    result, tr_velo_to_cam, r0_rect, P2, image_shape)
                result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
                lidar_bboxes = result_filter['lidar_bboxes']
                labels, scores = result_filter['labels'], result_filter['scores']
                bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
                for lidar_bbox, label, score, bbox2d, camera_bbox in \
                    zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
                    format_result['name'].append(LABEL2CLASSES[label])
                    format_result['truncated'].append(0.0)
                    format_result['occluded'].append(0)
                    alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
                    format_result['alpha'].append(alpha)
                    format_result['bbox'].append(bbox2d)
                    format_result['dimensions'].append(camera_bbox[3:6])
                    format_result['location'].append(camera_bbox[:3])
                    format_result['rotation_y'].append(camera_bbox[6])
                    format_result['score'].append(score)
                write_label(format_result, os.path.join(saved_submit_path, f'{idx:06d}.txt'))
                format_results[idx] = {k: np.array(v) for k, v in format_result.items()}
        write_pickle(format_results, os.path.join(saved_path, 'results.pkl'))
    print('\nAnalyzing empirical LIDAR detection errors...')
    error_analysis = error_analyzer.analyze_detection_errors(
        val_dataset.data_infos, format_results, CLASSES, args.iou_thresh
    )
    binned_errors = error_analyzer.calculate_distance_binned_errors(error_analysis)
    error_analyzer.print_error_summary(error_analysis, binned_errors)
    error_analyzer.save_error_analysis(error_analysis, binned_errors, saved_path)
    print(f'\nEmpirical LIDAR error analysis complete! Results saved to: {saved_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Empirical LIDAR Error Analysis Configuration')
    parser.add_argument('--data_root', default='./data/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', 
                        help='your checkpoint for kitti')
    parser.add_argument('--saved_path', default='error_analysis_results', 
                        help='your saved path for error analysis results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    parser.add_argument(
        '--iou_thresh', type=float, default=0.5, 
        help='IoU threshold for matching detections to ground truth'
    )
    args = parser.parse_args()
    main(args) 

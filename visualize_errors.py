import numpy as np
import matplotlib.pyplot as plt
import os
from pointpillars.utils import read_pickle


def plot_error_analysis(error_analysis_path, output_dir='error_plots'):
    """Plot empirical error analysis results."""
    # Load error analysis results
    results = read_pickle(error_analysis_path)
    error_analysis = results['error_analysis']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    fig_width = 12
    fig_height = 8
    
    # 1. Plot missed detection rates per class
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    classes = list(error_analysis['per_class'].keys())
    missed_rates = []
    class_names = []
    
    for class_name in classes:
        stats = error_analysis['per_class'][class_name]
        if stats['total_gt'] > 0:
            missed_rate = stats['missed_detections'] / stats['total_gt'] * 100
            missed_rates.append(missed_rate)
            class_names.append(class_name)
    
    bars = ax.bar(class_names, missed_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Missed Detection Rate (%)')
    ax.set_title('Missed Detection Rate by Class')
    ax.set_ylim(0, max(missed_rates) * 1.1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, missed_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missed_detection_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot empirical errors vs distance for each class
    error_types = ['distal', 'perp', 'height']
    error_labels = ['Distal Error (m)', 'Perpendicular Error (m)', 'Height Error (m)']
    
    for class_name in ['Car', 'Pedestrian', 'Cyclist']:
        if class_name not in error_analysis['per_class']:
            continue
            
        fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
        fig.suptitle(f'{class_name} - Empirical Errors vs Distance', fontsize=16)
        
        distances = np.array(error_analysis['per_class'][class_name]['distances'])
        
        for i, (error_type, error_label) in enumerate(zip(error_types, error_labels)):
            ax = axes[i]
            errors = np.array(error_analysis['per_class'][class_name][f'{error_type}_errors'])
            
            # Filter out NaN values (missed detections)
            valid_mask = ~np.isnan(errors)
            valid_distances = distances[valid_mask]
            valid_errors = np.abs(errors[valid_mask])  # Use absolute values
            
            if len(valid_errors) > 0:
                # Create scatter plot
                ax.scatter(valid_distances, valid_errors, alpha=0.6, s=20)
                
                # Add trend lines (linear and quadratic)
                if len(valid_errors) > 1:
                    # Linear fit
                    z_linear = np.polyfit(valid_distances, valid_errors, 1)
                    p_linear = np.poly1d(z_linear)
                    ax.plot(valid_distances, p_linear(valid_distances), "r--", alpha=0.8, linewidth=2, label='Linear')
                    
                    # Quadratic fit (if we have enough points)
                    if len(valid_errors) > 2:
                        z_quad = np.polyfit(valid_distances, valid_errors, 2)
                        p_quad = np.poly1d(z_quad)
                        ax.plot(valid_distances, p_quad(valid_distances), "g-", alpha=0.8, linewidth=2, label='Quadratic')
                    
                    # Add trend equations
                    slope = z_linear[0]
                    intercept = z_linear[1]
                    ax.text(0.05, 0.95, f'Linear: y = {slope:.4f}x + {intercept:.4f}', 
                           transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
                    
                    if len(valid_errors) > 2:
                        a, b, c = z_quad[0], z_quad[1], z_quad[2]
                        ax.text(0.05, 0.85, f'Quad: y = {a:.4f}x² + {b:.4f}x + {c:.4f}', 
                               transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
                    
                    ax.legend()
            
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel(error_label)
            ax.set_title(f'{class_name} {error_label}')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            if len(valid_errors) > 0:
                mean_error = np.mean(valid_errors)
                std_error = np.std(valid_errors)
                ax.text(0.05, 0.85, 
                       f'Mean: {mean_error:.4f} ± {std_error:.4f}', 
                       transform=ax.transAxes, 
                       bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{class_name.lower()}_errors_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Combined error comparison across classes
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    fig.suptitle('Error Comparison Across Classes', fontsize=16)
    
    for i, (error_type, error_label) in enumerate(zip(error_types, error_labels)):
        ax = axes[i]
        
        for class_name in ['Car', 'Pedestrian', 'Cyclist']:
            if class_name not in error_analysis['per_class']:
                continue
                
            distances = np.array(error_analysis['per_class'][class_name]['distances'])
            errors = np.array(error_analysis['per_class'][class_name][f'{error_type}_errors'])
            
            # Filter out NaN values
            valid_mask = ~np.isnan(errors)
            valid_distances = distances[valid_mask]
            valid_errors = np.abs(errors[valid_mask])  # Use absolute values
            
            if len(valid_errors) > 0:
                ax.scatter(valid_distances, valid_errors, alpha=0.6, s=15, label=class_name)
                
                # Add trend lines for each class
                if len(valid_errors) > 2:
                    # Linear fit
                    z_linear = np.polyfit(valid_distances, valid_errors, 1)
                    p_linear = np.poly1d(z_linear)
                    ax.plot(valid_distances, p_linear(valid_distances), 
                           alpha=0.6, linewidth=1, linestyle='--')
                    
                    # Quadratic fit
                    z_quad = np.polyfit(valid_distances, valid_errors, 2)
                    p_quad = np.poly1d(z_quad)
                    ax.plot(valid_distances, p_quad(valid_distances), 
                           alpha=0.6, linewidth=1, linestyle='-')
        
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel(error_label)
        ax.set_title(error_label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_comparison_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Error statistics summary table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Class', 'Total GT', 'Detected', 'Missed', 'Miss Rate (%)', 
               'Distal Mean±Std', 'Perp Mean±Std', 'Height Mean±Std']
    
    for class_name in ['Car', 'Pedestrian', 'Cyclist']:
        if class_name in error_analysis['per_class']:
            stats = error_analysis['per_class'][class_name]
            missed_rate = stats['missed_detections'] / stats['total_gt'] * 100 if stats['total_gt'] > 0 else 0
            
            # Calculate error statistics (using absolute values)
            distal_errors = np.abs(np.array(stats['distal_errors']))
            perp_errors = np.abs(np.array(stats['perp_errors']))
            height_errors = np.abs(np.array(stats['height_errors']))
            
            valid_distal = distal_errors[~np.isnan(distal_errors)]
            valid_perp = perp_errors[~np.isnan(perp_errors)]
            valid_height = height_errors[~np.isnan(height_errors)]
            
            distal_str = f"{np.mean(valid_distal):.4f}±{np.std(valid_distal):.4f}" if len(valid_distal) > 0 else "N/A"
            perp_str = f"{np.mean(valid_perp):.4f}±{np.std(valid_perp):.4f}" if len(valid_perp) > 0 else "N/A"
            height_str = f"{np.mean(valid_height):.4f}±{np.std(valid_height):.4f}" if len(valid_height) > 0 else "N/A"
            
            table_data.append([
                class_name,
                stats['total_gt'],
                stats['total_detections'],
                stats['missed_detections'],
                f'{missed_rate:.2f}',
                distal_str,
                perp_str,
                height_str
            ])
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Empirical Error Statistics Summary', fontsize=16, pad=20)
    plt.savefig(os.path.join(output_dir, 'error_statistics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Distance-binned error trends
    if 'binned_errors' in results:
        binned_errors = results['binned_errors']
        
        for class_name in ['Car', 'Pedestrian', 'Cyclist']:
            if class_name not in binned_errors['per_class']:
                continue
                
            fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
            fig.suptitle(f'{class_name} - Binned Error Trends', fontsize=16)
            
            distances = binned_errors['per_class'][class_name]['distance_bins']
            
            for i, (error_type, error_label) in enumerate(zip(['distal', 'perp', 'height'], 
                                                             ['Distal Error (m)', 'Perpendicular Error (m)', 'Height Error (m)'])):
                ax = axes[i]
                
                means = binned_errors['per_class'][class_name][f'{error_type}_mean']
                stds = binned_errors['per_class'][class_name][f'{error_type}_std']
                counts = binned_errors['per_class'][class_name]['counts']
                
                # Only plot bins with data
                valid_mask = np.array(counts) > 0
                if np.any(valid_mask):
                    valid_distances = np.array(distances)[valid_mask]
                    valid_means = np.abs(np.array(means)[valid_mask])  # Use absolute values
                    valid_stds = np.array(stds)[valid_mask]
                    
                    ax.errorbar(valid_distances, valid_means, yerr=valid_stds, 
                              fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
                    
                    # Add trend lines (linear and quadratic)
                    if len(valid_means) > 1:
                        # Linear fit
                        z_linear = np.polyfit(valid_distances, valid_means, 1)
                        p_linear = np.poly1d(z_linear)
                        ax.plot(valid_distances, p_linear(valid_distances), "r--", alpha=0.8, linewidth=2, label='Linear')
                        
                        # Quadratic fit (if we have enough points)
                        if len(valid_means) > 2:
                            z_quad = np.polyfit(valid_distances, valid_means, 2)
                            p_quad = np.poly1d(z_quad)
                            ax.plot(valid_distances, p_quad(valid_distances), "g-", alpha=0.8, linewidth=2, label='Quadratic')
                        
                        ax.legend()
                
                ax.set_xlabel('Distance (m)')
                ax.set_ylabel(error_label)
                ax.set_title(f'{class_name} {error_label}')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{class_name.lower()}_binned_errors.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 6. Missed detection rate vs distance
    if 'binned_errors' in results:
        binned_errors = results['binned_errors']
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        markers = ['o', 's', '^']
        
        for i, class_name in enumerate(['Car', 'Pedestrian', 'Cyclist']):
            if class_name not in binned_errors['per_class']:
                continue
                
            distances = binned_errors['per_class'][class_name]['distance_bins']
            counts = binned_errors['per_class'][class_name]['counts']
            missed = binned_errors['per_class'][class_name]['missed']
            
            # Calculate missed detection rate for each distance bin
            missed_rates = []
            valid_distances = []
            
            for j, (count, missed_count) in enumerate(zip(counts, missed)):
                if count > 0:  # Only include bins with ground truth objects
                    missed_rate = (missed_count / count) * 100
                    missed_rates.append(missed_rate)
                    valid_distances.append(distances[j])
            
            if len(missed_rates) > 0:
                ax.plot(valid_distances, missed_rates, 
                       marker=markers[i], color=colors[i], linewidth=2, markersize=8, 
                       label=class_name, alpha=0.8)
                
                # Add trend lines (linear and quadratic)
                if len(missed_rates) > 1:
                    # Linear fit
                    z_linear = np.polyfit(valid_distances, missed_rates, 1)
                    p_linear = np.poly1d(z_linear)
                    ax.plot(valid_distances, p_linear(valid_distances), 
                           color=colors[i], linestyle='--', alpha=0.6, linewidth=1)
                    
                    # Quadratic fit (if we have enough points)
                    if len(missed_rates) > 2:
                        z_quad = np.polyfit(valid_distances, missed_rates, 2)
                        p_quad = np.poly1d(z_quad)
                        ax.plot(valid_distances, p_quad(valid_distances), 
                               color=colors[i], linestyle='-', alpha=0.6, linewidth=1)
        
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Missed Detection Rate (%)')
        ax.set_title('Missed Detection Rate vs Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missed_detection_vs_distance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Individual missed detection rate plots per class
        for class_name in ['Car', 'Pedestrian', 'Cyclist']:
            if class_name not in binned_errors['per_class']:
                continue
                
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            distances = binned_errors['per_class'][class_name]['distance_bins']
            counts = binned_errors['per_class'][class_name]['counts']
            missed = binned_errors['per_class'][class_name]['missed']
            
            # Calculate missed detection rate for each distance bin
            missed_rates = []
            valid_distances = []
            
            for j, (count, missed_count) in enumerate(zip(counts, missed)):
                if count > 0:  # Only include bins with ground truth objects
                    missed_rate = (missed_count / count) * 100
                    missed_rates.append(missed_rate)
                    valid_distances.append(distances[j])
            
            if len(missed_rates) > 0:
                ax.plot(valid_distances, missed_rates, 
                       marker='o', color='#1f77b4', linewidth=2, markersize=8, alpha=0.8)
                
                # Add trend lines (linear and quadratic)
                if len(missed_rates) > 1:
                    # Linear fit
                    z_linear = np.polyfit(valid_distances, missed_rates, 1)
                    p_linear = np.poly1d(z_linear)
                    ax.plot(valid_distances, p_linear(valid_distances), 
                           color='red', linestyle='--', alpha=0.8, linewidth=2, label='Linear')
                    
                    # Quadratic fit (if we have enough points)
                    if len(missed_rates) > 2:
                        z_quad = np.polyfit(valid_distances, missed_rates, 2)
                        p_quad = np.poly1d(z_quad)
                        ax.plot(valid_distances, p_quad(valid_distances), 
                               color='green', linestyle='-', alpha=0.8, linewidth=2, label='Quadratic')
                    
                    # Add trend equations
                    slope = z_linear[0]
                    intercept = z_linear[1]
                    ax.text(0.05, 0.95, f'Linear: y = {slope:.2f}x + {intercept:.2f}', 
                           transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
                    
                    if len(missed_rates) > 2:
                        a, b, c = z_quad[0], z_quad[1], z_quad[2]
                        ax.text(0.05, 0.85, f'Quad: y = {a:.2f}x² + {b:.2f}x + {c:.2f}', 
                               transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
                    
                    ax.legend()
                
                # Add statistics
                mean_rate = np.mean(missed_rates)
                ax.text(0.05, 0.85, f'Mean Miss Rate: {mean_rate:.1f}%', 
                       transform=ax.transAxes, 
                       bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Missed Detection Rate (%)')
            ax.set_title(f'{class_name} - Missed Detection Rate vs Distance')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{class_name.lower()}_missed_detection_vs_distance.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Visualization complete! Plots saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Empirical Error Analysis Results')
    parser.add_argument('--error_analysis_path', default='error_analysis_results/error_analysis.pkl',
                        help='Path to error analysis results file')
    parser.add_argument('--output_dir', default='error_plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.error_analysis_path):
        print(f"Error: {args.error_analysis_path} not found!")
        print("Please run evaluate_errors.py first to generate error analysis results.")
    else:
        plot_error_analysis(args.error_analysis_path, args.output_dir) 

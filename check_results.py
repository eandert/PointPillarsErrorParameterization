import pickle

# Load evaluation results
with open('error_predictor_evaluation_100epochs/evaluation_results.pkl', 'rb') as f:
    data = pickle.load(f)

print("Evaluation Results:")
print("=" * 50)

if data['nn_metrics']:
    print("Neural Network:")
    for target, metrics in data['nn_metrics'].items():
        print(f"  {target}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

print("\nLinear/Quadratic:")
for target, metrics in data['linear_metrics'].items():
    print(f"  {target}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

# Load training metrics
with open('error_predictor_results_100epochs/training_metrics.pkl', 'rb') as f:
    train_data = pickle.load(f)

print("\nTraining Results:")
print("=" * 50)
print(f"Final Train Loss: {train_data['train_losses'][-1]:.4f}")
print(f"Final Val Loss: {train_data['val_losses'][-1]:.4f}")
print(f"Best Val Loss: {min(train_data['val_losses']):.4f}")

if 'val_metrics' in train_data:
    print("\nTraining R² Scores:")
    for target, metrics in train_data['val_metrics'].items():
        print(f"  {target}: {metrics['R2']:.4f}") 

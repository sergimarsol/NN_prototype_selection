"""
Plot prototype selection results across different M values.

This script generates a line plot showing the accuracy of different prototype
selection methods (Run 1-5 + Random Stratified) as a function of the prototype
budget M.

Features:
- Line plot with different colors for each method
- Error bars for the randomized method (Random Stratified)
- Legend showing all methods
- Grid for readability
- Professional formatting

Usage:
    python plot_results.py
    
Output:
    results_plot.png (saved in current directory)
"""

import numpy as np
import matplotlib.pyplot as plt

# Data: M values and accuracies for each method
M_values = np.array([100, 200, 500, 1000, 2000, 5000, 10000])

# Random Stratified: mean ± std (will plot as lines with error bars)
random_stratified = np.array([0.6555, 0.7335, 0.7980, 0.8388, 0.8650, 0.8972, 0.9146])
random_stratified_std = np.array([0.0202, 0.0120, 0.0062, 0.0035, 0.0035, 0.0027, 0.0026])

# Equal K-Means
run1_equal_kmeans = np.array([0.7962, 0.8206, 0.8553, 0.8749, 0.8972, 0.9091, 0.9187])

# Error-Driven
run2_error_driven = np.array([0.7962, 0.8206, 0.8555, 0.8849, 0.9030, 0.9141, 0.9242])

# Iterative Error
run3_iterative_error = np.array([0.7918, 0.8191, 0.8619, 0.8855, 0.8995, 0.9158, 0.9247])

# Selective Hybrid
run5_selective_hybrid = np.array([0.7915, 0.8177, 0.8591, 0.8833, 0.8977, 0.9143, 0.9226])


# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 7))

# Plot each method with distinct colors
colors = {
    'Random Stratified': '#E74C3C',  # Red
    'Equal K-Means': '#3498DB',  # Blue
    'Error-Driven': '#2ECC71',  # Green
    'Iterative Error': '#F39C12',  # Orange
    'Selective Hybrid': '#9B59B6',  # Purple
}

linewidth = 2.5
markersize = 8

# Plot Random Stratified with error bars
ax.errorbar(M_values, random_stratified, yerr=random_stratified_std, 
            label='Random Stratified', color=colors['Random Stratified'],
            linewidth=linewidth, marker='o', markersize=markersize,
            capsize=5, capthick=2, alpha=0.8)

# Plot other methods without error bars
ax.plot(M_values, run1_equal_kmeans, label='Equal K-Means',
        color=colors['Equal K-Means'], linewidth=linewidth,
        marker='s', markersize=markersize, alpha=0.8)

ax.plot(M_values, run2_error_driven, label='Error-Driven',
        color=colors['Error-Driven'], linewidth=linewidth,
        marker='^', markersize=markersize, alpha=0.8)

ax.plot(M_values, run3_iterative_error, label='Iterative Error',
        color=colors['Iterative Error'], linewidth=linewidth,
        marker='D', markersize=markersize, alpha=0.8)

ax.plot(M_values, run5_selective_hybrid, label='Selective Hybrid',
        color=colors['Selective Hybrid'], linewidth=linewidth,
        marker='*', markersize=markersize+4, alpha=0.8)


# Customize the plot
ax.set_xlabel('Number of Prototypes (M)', fontsize=12, fontweight='bold')
ax.set_ylabel('1-NN Test Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Prototype Selection Methods: Accuracy vs. Budget Size (MNIST)', 
             fontsize=14, fontweight='bold', pad=20)

# Set x-axis to log scale for better visualization
ax.set_xscale('log')

# Add grid for readability
ax.grid(True, alpha=0.3, linestyle='--')

# Set y-axis limits with some padding
ax.set_ylim([0.6, 0.95])

# Add legend
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)

# Tight layout
plt.tight_layout()

# Save the plot
output_file = 'results_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_file}")

# Also display the plot
plt.show()


# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

methods = {
    'Random Stratified': random_stratified,
    'Equal K-Means': run1_equal_kmeans,
    'Error-Driven': run2_error_driven,
    'Iterative Error': run3_iterative_error,
    'Selective Hybrid': run5_selective_hybrid,
}

for method_name, accuracies in methods.items():
    improvement = accuracies[-1] - accuracies[0]
    percent_improvement = (improvement / accuracies[0]) * 100
    print(f"\n{method_name}:")
    print(f"  M=100 accuracy: {accuracies[0]:.4f}")
    print(f"  M=10000 accuracy: {accuracies[-1]:.4f}")
    print(f"  Total improvement: {improvement:+.4f} ({percent_improvement:+.2f}%)")
    print(f"  Mean accuracy: {np.mean(accuracies):.4f}")

# Best method at each M
print("\n" + "="*70)
print("BEST METHOD AT EACH M VALUE")
print("="*70)

for i, m in enumerate(M_values):
    accuracies_at_m = {
        'Random Stratified': random_stratified[i],
        'Equal K-Means': run1_equal_kmeans[i],
        'Error-Driven': run2_error_driven[i],
        'Iterative Error': run3_iterative_error[i],
        'Selective Hybrid': run5_selective_hybrid[i],
    }
    best_method = max(accuracies_at_m, key=accuracies_at_m.get)
    best_acc = accuracies_at_m[best_method]
    print(f"M={m:5d}: {best_method:30s} ({best_acc:.4f})")

"""
Plot per-class accuracy for different prototype selection methods (M=10000 and M=100).

This script generates a line plot showing the per-class accuracy of different 
prototype selection methods across the 10 MNIST digit classes.

Features:
- Solid lines for M=10000, dotted lines for M=100
- Error bars for the randomized method (Random Stratified)
- Legend showing all methods
- Grid for readability
- Professional formatting

Usage:
    python plot_per_class_results.py
    
Output:
    per_class_results_plot.png (saved in current directory)
"""

import numpy as np
import matplotlib.pyplot as plt

# Data: Classes 0-9 and per-class accuracies for each method
classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# M=10000 (solid lines)
random_stratified_10000 = np.array([0.9750, 0.9911, 0.8884, 0.9120, 0.9047, 0.8868, 0.9506, 0.8923, 0.8523, 0.8824])
random_stratified_std_10000 = np.array([0.0043, 0.0019, 0.0090, 0.0083, 0.0107, 0.0106, 0.0051, 0.0049, 0.0123, 0.0070])

run1_equal_kmeans_10000 = np.array([0.9806, 0.9965, 0.8721, 0.9059, 0.9114, 0.8991, 0.9551, 0.8930, 0.8706, 0.8940])
run2_error_driven_10000 = np.array([0.9500, 0.9806, 0.9060, 0.9089, 0.9216, 0.9002, 0.9551, 0.8930, 0.9055, 0.9138])
run3_iterative_error_10000 = np.array([0.9612, 0.9930, 0.8876, 0.9040, 0.9206, 0.9036, 0.9676, 0.9066, 0.8973, 0.8979])
run5_selective_hybrid_10000 = np.array([0.9643, 0.9868, 0.8973, 0.8960, 0.9257, 0.9013, 0.9541, 0.9115, 0.8932, 0.8880])

# M=100 (dotted lines)
random_stratified_100 = np.array([0.8060, 0.9797, 0.4624, 0.6134, 0.5875, 0.5642, 0.7089, 0.5524, 0.6133, 0.6260])
random_stratified_std_100 = np.array([0.0576, 0.0103, 0.0949, 0.0954, 0.0730, 0.0708, 0.0601, 0.0884, 0.0556, 0.1357])

run1_equal_kmeans_100 = np.array([0.9367, 0.9921, 0.6734, 0.7515, 0.8014, 0.7567, 0.8079, 0.7539, 0.7290, 0.7364])
run2_error_driven_100 = np.array([0.9367, 0.9921, 0.6734, 0.7515, 0.8014, 0.7567, 0.8079, 0.7539, 0.7290, 0.7364])
run3_iterative_error_100 = np.array([0.9571, 0.9947, 0.6734, 0.7396, 0.7821, 0.6850, 0.8319, 0.7228, 0.7546, 0.7483])
run5_selective_hybrid_100 = np.array([0.9367, 0.9639, 0.6773, 0.7505, 0.7821, 0.7522, 0.8058, 0.7403, 0.7320, 0.7542])


# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 7))

# Color scheme for methods
colors = {
    'Random Stratified': '#E74C3C',  # Red
    'Equal K-Means': '#3498DB',  # Blue
    'Error-Driven': '#2ECC71',  # Green
    'Iterative Error': '#F39C12',  # Orange
    'Selective Hybrid': '#9B59B6',  # Purple
}

linewidth = 2.5
markersize = 8

# Plot M=10000 (solid)
ax.errorbar(classes, random_stratified_10000, yerr=random_stratified_std_10000,
            label='Random Stratified (M=10000)', color=colors['Random Stratified'],
            linewidth=linewidth, marker='o', markersize=markersize,
            capsize=5, capthick=2, alpha=0.8)

# Plot other methods without error bars
ax.plot(classes, run1_equal_kmeans_10000, label='Equal K-Means (M=10000)',
        color=colors['Equal K-Means'], linewidth=linewidth,
        marker='s', markersize=markersize, alpha=0.8)

ax.plot(classes, run2_error_driven_10000, label='Error-Driven (M=10000)',
        color=colors['Error-Driven'], linewidth=linewidth,
        marker='^', markersize=markersize, alpha=0.8)

ax.plot(classes, run3_iterative_error_10000, label='Iterative Error (M=10000)',
        color=colors['Iterative Error'], linewidth=linewidth,
        marker='D', markersize=markersize, alpha=0.8)

ax.plot(classes, run5_selective_hybrid_10000, label='Selective Hybrid (M=10000)',
        color=colors['Selective Hybrid'], linewidth=linewidth,
        marker='*', markersize=markersize+4, alpha=0.8)

# Plot M=100 (dotted)
ax.errorbar(classes, random_stratified_100, yerr=random_stratified_std_100,
        label='Random Stratified (M=100)', color=colors['Random Stratified'],
        linewidth=linewidth, linestyle=':', marker='o', markersize=markersize-1,
        capsize=5, capthick=2, alpha=0.5)

ax.plot(classes, run1_equal_kmeans_100, label='Equal K-Means (M=100)',
    color=colors['Equal K-Means'], linewidth=linewidth, linestyle=':',
    marker='s', markersize=markersize-1, alpha=0.5)

ax.plot(classes, run2_error_driven_100, label='Error-Driven (M=100)',
    color=colors['Error-Driven'], linewidth=linewidth, linestyle=':',
    marker='^', markersize=markersize-1, alpha=0.5)

ax.plot(classes, run3_iterative_error_100, label='Iterative Error (M=100)',
    color=colors['Iterative Error'], linewidth=linewidth, linestyle=':',
    marker='D', markersize=markersize-1, alpha=0.5)

ax.plot(classes, run5_selective_hybrid_100, label='Selective Hybrid (M=100)',
    color=colors['Selective Hybrid'], linewidth=linewidth, linestyle=':',
    marker='*', markersize=markersize+2, alpha=0.5)


# Customize the plot
ax.set_xlabel('MNIST Digit Class', fontsize=12, fontweight='bold')
ax.set_ylabel('1-NN Test Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Accuracy for Different Prototype Selection Methods (M=10000 vs M=100)',
             fontsize=14, fontweight='bold', pad=20)

# Set x-axis to show all classes
ax.set_xticks(classes)
ax.set_xticklabels([f'{i}' for i in classes])

# Add grid for readability
ax.grid(True, alpha=0.3, linestyle='--')

# Set y-axis limits
ax.set_ylim([0.45, 1.00])

# Add legend
ax.legend(loc='lower left', fontsize=10, framealpha=0.95, ncol=2)

# Tight layout
plt.tight_layout()

# Save the plot
output_file = 'per_class_results_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_file}")

# Also display the plot
plt.show()


# Print summary statistics
print("\n" + "="*80)
print("PER-CLASS ACCURACY SUMMARY (M=10000)")
print("="*80)

methods = {
    'Random Stratified': random_stratified_10000,
    'Equal K-Means': run1_equal_kmeans_10000,
    'Error-Driven': run2_error_driven_10000,
    'Iterative Error': run3_iterative_error_10000,
    'Selective Hybrid': run5_selective_hybrid_10000,
}

# Print per-method statistics
for method_name, accuracies in methods.items():
    print(f"\n{method_name}:")
    print(f"  Mean accuracy:   {np.mean(accuracies):.4f}")
    print(f"  Min accuracy:    {np.min(accuracies):.4f} (Class {np.argmin(accuracies)})")
    print(f"  Max accuracy:    {np.max(accuracies):.4f} (Class {np.argmax(accuracies)})")
    print(f"  Std deviation:   {np.std(accuracies):.4f}")

# Identify hardest and easiest classes for each method
print("\n" + "="*80)
print("HARDEST AND EASIEST CLASSES PER METHOD")
print("="*80)

for method_name, accuracies in methods.items():
    hardest_class = np.argmin(accuracies)
    easiest_class = np.argmax(accuracies)
    print(f"\n{method_name}:")
    print(f"  Hardest class: {hardest_class} ({accuracies[hardest_class]:.4f})")
    print(f"  Easiest class: {easiest_class} ({accuracies[easiest_class]:.4f})")

# Find best method per class
print("\n" + "="*80)
print("BEST METHOD PER CLASS")
print("="*80)

for class_idx in classes:
    accuracies_per_method = {
        'Random Stratified': random_stratified_10000[class_idx],
        'Equal K-Means': run1_equal_kmeans_10000[class_idx],
        'Error-Driven': run2_error_driven_10000[class_idx],
        'Iterative Error': run3_iterative_error_10000[class_idx],
        'Selective Hybrid': run5_selective_hybrid_10000[class_idx],
    }
    best_method = max(accuracies_per_method, key=accuracies_per_method.get)
    best_acc = accuracies_per_method[best_method]
    worst_method = min(accuracies_per_method, key=accuracies_per_method.get)
    worst_acc = accuracies_per_method[worst_method]
    
    print(f"\nClass {class_idx}:")
    print(f"  Best:  {best_method:30s} ({best_acc:.4f})")
    print(f"  Worst: {worst_method:30s} ({worst_acc:.4f})")

# Overall comparison
print("\n" + "="*80)
print("OVERALL METHOD RANKING (by mean per-class accuracy)")
print("="*80)

method_means = {name: np.mean(accs) for name, accs in methods.items()}
sorted_methods = sorted(method_means.items(), key=lambda x: x[1], reverse=True)

for rank, (method_name, mean_acc) in enumerate(sorted_methods, 1):
    print(f"{rank}. {method_name:30s} {mean_acc:.4f}")

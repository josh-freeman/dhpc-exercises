#!/usr/bin/env python3
"""
Plotting script for 2D heat diffusion output.
Combines output from multiple MPI ranks into a single visualization.

Usage: python3 plot_heat.py

Expects files: heat_output_0.dat, heat_output_1.dat, etc.
"""
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys

def read_data_files():
    """Read all heat output files and combine into a single grid."""
    files = sorted(glob.glob('heat_output_*.dat'))

    if not files:
        print("No heat_output_*.dat files found!")
        print("Run the simulation first:")
        print("  mpicc -O2 -o step4 step4.c -lm")
        print("  mpirun -np 4 ./step4")
        sys.exit(1)

    print(f"Found {len(files)} output files")

    # Read all data points
    all_data = []
    for fname in files:
        with open(fname, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) == 3:
                    x, y, val = int(parts[0]), int(parts[1]), float(parts[2])
                    all_data.append((x, y, val))

    if not all_data:
        print("No data found in files!")
        sys.exit(1)

    # Determine grid size
    max_x = max(d[0] for d in all_data) + 1
    max_y = max(d[1] for d in all_data) + 1

    print(f"Grid size: {max_x} x {max_y}")

    # Create 2D array
    grid = np.zeros((max_x, max_y))
    for x, y, val in all_data:
        grid[x, y] = val

    return grid


def plot_heat(grid, output_file='heat_diffusion.png'):
    """Create a visualization of the temperature field."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: 2D heatmap
    im = axes[0].imshow(grid, cmap='hot', origin='lower', aspect='equal')
    axes[0].set_xlabel('Y position', fontsize=12)
    axes[0].set_ylabel('X position', fontsize=12)
    axes[0].set_title('2D Heat Diffusion - Temperature Field', fontsize=14)
    plt.colorbar(im, ax=axes[0], label='Temperature')

    # Plot 2: Cross-section through center
    center_x = grid.shape[0] // 2
    axes[1].plot(grid[center_x, :], 'b-', linewidth=2, label=f'Row {center_x}')

    center_y = grid.shape[1] // 2
    axes[1].plot(grid[:, center_y], 'r--', linewidth=2, label=f'Column {center_y}')

    axes[1].set_xlabel('Position', fontsize=12)
    axes[1].set_ylabel('Temperature', fontsize=12)
    axes[1].set_title('Temperature Cross-sections', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")

    # Print statistics
    print(f"\n=== Statistics ===")
    print(f"Min temperature: {grid.min():.4f}")
    print(f"Max temperature: {grid.max():.4f}")
    print(f"Mean temperature: {grid.mean():.4f}")


def plot_3d(grid, output_file='heat_diffusion_3d.png'):
    """Create a 3D surface plot."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Downsample for smoother visualization
    step = max(1, grid.shape[0] // 50)
    X = np.arange(0, grid.shape[0], step)
    Y = np.arange(0, grid.shape[1], step)
    X, Y = np.meshgrid(X, Y)
    Z = grid[::step, ::step].T

    surf = ax.plot_surface(X, Y, Z, cmap='hot', linewidth=0, antialiased=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature')
    ax.set_title('2D Heat Diffusion - 3D View')
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.savefig(output_file, dpi=150)
    print(f"3D plot saved to {output_file}")


if __name__ == '__main__':
    grid = read_data_files()
    plot_heat(grid)

    # Try 3D plot
    try:
        plot_3d(grid)
    except Exception as e:
        print(f"Note: 3D plot failed ({e}), continuing...")

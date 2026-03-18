import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import numpy as np

def predictions_vs_observations(y, y_pred):

    # Convert to NumPy for plotting
    y_np = y.detach().numpy()
    y_pred_np = y_pred.detach().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_np, y_pred_np, color='blue', label='Predictions')
    plt.plot([y_np.min(), y_np.max()], [y_np.min(), y_np.max()], 'r--', label='Ideal fit (y = x)')
    plt.xlabel("Observed values")
    plt.ylabel("Predicted values")
    plt.title("Predictions vs Observations")
    plt.legend()
    plt.grid(True)
    plt.show()
    return True



def feature_plotting(X, B_star, y):
    # Predictions
    y_pred = X @ B_star

    x1 = X[:, 1].detach().numpy()
    x2 = X[:, 2].detach().numpy()
    y_np = y.detach().numpy()
    y_pred_np = y_pred.detach().numpy()
    B = B_star.detach().numpy()

    x1_range = np.linspace(x1.min(), x1.max(), 30)
    x2_range = np.linspace(x2.min(), x2.max(), 30)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    y_grid = B[0] + B[1] * x1_grid + B[2] * x2_grid 

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Observed and predicted points
    ax.scatter(x1, x2, y_np, color='blue', label='Observed', s=40)
    ax.scatter(x1, x2, y_pred_np, color='red', marker='^', label='Predicted', s=40)

    ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.4, color='orange')

    plane_patch = mpatches.Patch(color='orange', alpha=0.4, label='Regression Plane')
    ax.legend(handles=[
        plt.Line2D([], [], color='blue', marker='o', linestyle='None', label='Observed'),
        plt.Line2D([], [], color='red', marker='^', linestyle='None', label='Predicted'),
        plane_patch
    ])

    ax.set_xlabel('Feature 1 (X₁)')
    ax.set_ylabel('Feature 2 (X₂)')
    ax.set_zlabel('Target / Prediction')
    ax.set_title('3D: Observed & Predicted with Regression Plane')

    for angle in [25, 45, 75, 120]:
        ax.view_init(elev=20, azim=angle)
        plt.pause(0.5)

    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(x1, y_np, color='blue', label='Observed')
    axes[0].scatter(x1, y_pred_np, color='orange', marker='^', label='Predicted')
    axes[0].set_xlabel('Feature 1 (X₁)')
    axes[0].set_ylabel('Target')
    axes[0].set_title('Feature 1 vs Target')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].scatter(x2, y_np, color='blue', label='Observed')
    axes[1].scatter(x2, y_pred_np, color='orange', marker='^', label='Predicted')
    axes[1].set_xlabel('Feature 2 (X₂)')
    axes[1].set_ylabel('Target')
    axes[1].set_title('Feature 2 vs Target')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(15, 5))

    # Default angle
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(x1, x2, y_np, color='blue', s=40)
    ax1.scatter(x1, x2, y_pred_np, color='red', marker='^', s=40)
    ax1.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.4, color='orange')
    ax1.set_title('3D View 1 (Default)')
    ax1.set_xlabel('X₁'); ax1.set_ylabel('X₂'); ax1.set_zlabel('Target')

    # Different angle 1
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(x1, x2, y_np, color='blue', s=40)
    ax2.scatter(x1, x2, y_pred_np, color='red', marker='^', s=40)
    ax2.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.4, color='orange')
    ax2.view_init(elev=25, azim=45)
    ax2.set_title('3D View 2')

    # Different angle 2
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(x1, x2, y_np, color='blue', s=40)
    ax3.scatter(x1, x2, y_pred_np, color='red', marker='^', s=40)
    ax3.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.4, color='orange')
    ax3.view_init(elev=10, azim=120)
    ax3.set_title('3D View 3')

    plt.tight_layout()
    plt.show()
    return 0

def plot_residuals(y, y_pred, X):
    residuals = y_pred - y

    # Convert to NumPy
    x1 = X[:, 1].detach().numpy()
    x2 = X[:, 2].detach().numpy()
    y_np = y.detach().numpy()
    y_pred_np = y_pred.detach().numpy()
    resid_np = residuals.detach().numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x1, x2, resid_np, c=resid_np, cmap='coolwarm', s=60)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel('Feature 1 (X₁)')
    ax.set_ylabel('Feature 2 (X₂)')
    ax.set_zlabel('Residual (y_pred - y)')
    ax.set_title('3D Residuals Plot')
    plt.colorbar(sc, ax=ax, label='Residual Value')

    plt.show()


    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred_np, resid_np, color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals (y_pred - y)')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Residuals vs X1
    axes[0].scatter(x1, resid_np, color='teal')
    axes[0].axhline(0, color='black', linestyle='--')
    axes[0].set_xlabel('Feature 1 (X₁)')
    axes[0].set_ylabel('Residual (y_pred - y)')
    axes[0].set_title('Residuals vs Feature 1')
    axes[0].grid(True)

    # Residuals vs X2
    axes[1].scatter(x2, resid_np, color='darkorange')
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set_xlabel('Feature 2 (X₂)')
    axes[1].set_ylabel('Residual (y_pred - y)')
    axes[1].set_title('Residuals vs Feature 2')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    return 0
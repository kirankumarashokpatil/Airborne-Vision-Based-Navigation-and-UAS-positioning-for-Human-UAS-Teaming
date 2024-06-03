import cv2
import numpy as np
import orb_slam3
import torch
from slam.monocular_slam import MonocularSLAM
from datasets.tartanair_loader import load_tartanair_dataset
from models.autoencoder import train_autoencoder
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def calculate_rmse(estimated_poses, ground_truth_poses):
    errors = []
    for est, gt in zip(estimated_poses, ground_truth_poses):
        error = np.linalg.norm(est[:3, 3] - gt[:3, 3])
        errors.append(error)
    return np.sqrt(np.mean(np.square(errors)))

def update_plot(i, frames, slam, scatter, ax1, ax2):
    frame = frames[i]
    processed_frame = slam.process_frame(frame)
    
    ax1.clear()
    ax1.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    
    poses = np.array([pose[:3, 3] for pose in slam.estimated_poses])
    if len(poses) > 0:
        scatter.set_offsets(poses[:, :2])
        ax2.set_xlim(np.min(poses[:, 0]) - 1, np.max(poses[:, 0]) + 1)
        ax2.set_ylim(np.min(poses[:, 1]) - 1, np.max(poses[:, 1]) + 1)
    return scatter,

def main():
    vocab_file = "ORBvoc.txt"
    settings_file = "settings.yaml"
    orb_slam = orb_slam3.ORBSLAM3(vocab_file, settings_file)

    env_path = 'datasets'
    traj_path = 'P006'

    frames, ground_truth_poses = load_tartanair_dataset(env_path, traj_path)
    autoencoder_model = train_autoencoder(frames)
    slam = MonocularSLAM()

    slam.estimated_poses = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    scatter = ax2.scatter([], [])

    def update(i):
        frame = frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        orb_slam.TrackMonocular(gray, i / 10.0)  # Using frame index as timestamp
        processed_frame = slam.process_frame(frame)
        
        ax1.clear()
        ax1.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        
        poses = np.array([pose[:3, 3] for pose in slam.estimated_poses])
        if len(poses) > 0:
            scatter.set_offsets(poses[:, :2])
            ax2.set_xlim(np.min(poses[:, 0]) - 1, np.max(poses[:, 0]) + 1)
            ax2.set_ylim(np.min(poses[:, 1]) - 1, np.max(poses[:, 1]) + 1)
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50)
    plt.show()

    if slam.estimated_poses:
        rmse = calculate_rmse(slam.estimated_poses, ground_truth_poses)
        print(f'RMSE: {rmse:.4f}')
    else:
        print("No estimated poses to calculate RMSE")

    orb_slam.Shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

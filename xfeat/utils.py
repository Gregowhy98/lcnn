import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def draw_keypoints_on_image(image, keypoints):
    keypoints = [cv2.KeyPoint(x[0], x[1], 2) for x in keypoints.cpu().numpy()]
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    return img_with_keypoints

def draw_scores_heatmap(keypoints, scores):
    scores = scores.cpu().numpy()
    heatmap, xedges, yedges = np.histogram2d(keypoints[:, 0].cpu().numpy(), keypoints[:, 1].cpu().numpy(), bins=(120, 160), weights=scores)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', vmin=np.min(scores), vmax=np.max(scores))
    plt.colorbar()
    plt.title('Keypoint Scores Heatmap')
    return plt
    
def visualize_descriptors(descriptors, scores):
    descriptors = descriptors.cpu().numpy()
    pca = PCA(n_components=2)
    descriptors_2d = pca.fit_transform(descriptors)
    plt.figure(figsize=(8, 6))
    plt.scatter(descriptors_2d[:, 0], descriptors_2d[:, 1], c=scores, cmap='viridis')
    plt.colorbar()
    plt.title('Descriptors PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    return plt
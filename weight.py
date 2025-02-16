import torch
import torch.nn as nn
import matplotlib.pyplot as plt

loaded_model = torch.jit.load('grendel_octree_test/point_cloud/iteration_59997/_rk1color_mlp.pt')


weights = []
for name, param in loaded_model.named_parameters():
    if 'weight' in name:
        weights.append(param)

# 可视化权重
for i, weight in enumerate(weights):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f'Weight Layer {i+1} - Heatmap')
    plt.imshow(weight.detach().cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title(f'Weight Layer {i+1} - Histogram')
    plt.hist(weight.detach().cpu().numpy().flatten(), bins=50)
    plt.show()
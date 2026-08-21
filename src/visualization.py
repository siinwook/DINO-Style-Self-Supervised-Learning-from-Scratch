import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def visualize_attention_map(model, image, idx, device):
  img = image.unsqueeze(0).to(device)

  model.to(device)
  model.eval()

  last_attention = model.Encoder.EncBlcockList[-1].Attention
  last_attention.save_attention = True

  with torch.no_grad():
    out = model(img)

  attention_map = last_attention.attention_map[:,:,0,:].squeeze(0)
  image_np = image.permute(1,2,0).cpu().numpy()

  fig, axes = plt.subplots(3,3,figsize=(15,15),dpi=120)
  axes = axes.flatten()

  axes[0].axis("off")
  axes[1].imshow(image_np)
  axes[1].set_title(f"Original: STL-10 test index {idx}",fontsize=15)
  axes[1].axis("off")
  axes[2].axis("off")

  for head in range(6):
    head_values, head_indices = attention_map[head][1:].sort(descending=True)

    selected_indices = []
    mass = 0.0
    for i in range(len(head_values)):
      mass += head_values[i].item()
      selected_indices.append(head_indices[i].item())
      if mass >= 0.6:
        break

    patch_mask = torch.zeros(14*14)
    patch_mask[torch.tensor(selected_indices)] = 1
    patch_mask = patch_mask.reshape(1,1,14,14)

    pixel_mask = F.interpolate(patch_mask,size=image.shape[-2:],mode="nearest")[0,0]
    mask_np = pixel_mask.cpu().numpy()

    axes[head+3].imshow(image_np)
    axes[head+3].imshow(mask_np,alpha=0.45,vmin=0,vmax=1)
    axes[head+3].set_title(f"Head {head+1} ({len(selected_indices)} patches)",fontsize=15)
    axes[head+3].axis("off")

  plt.tight_layout()
  plt.show()

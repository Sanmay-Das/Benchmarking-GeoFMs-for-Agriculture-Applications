import torch

ckpt = torch.load('/bigdata/eldawylab/sdas050/MS_Research/weights/pretrain-vit-large-e199.pth', map_location='cpu')

print("Checkpoint keys:", ckpt.keys())
print("\nModel state_dict keys (first 20):")
for i, k in enumerate(list(ckpt['model'].keys())[:20]):
    print(f"  {k}: {ckpt['model'][k].shape}")

# Check for GroupC-specific keys
model_keys = ckpt['model'].keys()
has_channel_embed = any('channel_embed' in k for k in model_keys)
has_blocks_channel = any('blocks' in k and 'channel' in k for k in model_keys)

print(f"\nHas channel_embed keys: {has_channel_embed}")
print(f"Has channel-related block keys: {has_blocks_channel}")

# Check patch_embed shape
if 'patch_embed.proj.weight' in model_keys:
    shape = ckpt['model']['patch_embed.proj.weight'].shape
    print(f"\npatch_embed.proj.weight shape: {shape}")
    print(f"  -> Embedding dim: {shape[0]}, Channels: {shape[1]}, Patch: {shape[2]}x{shape[3]}")
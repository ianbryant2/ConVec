import torch


def normalize(tensor, mask, min_val, max_val,dim):
    assert tensor.shape == mask.shape

    result_tensor = tensor.clone()

    flattened_tensor = tensor.view(-1, tensor.shape[dim])
    flattened_mask = mask.view(-1, mask.shape[dim])

    valid_mask = flattened_mask == 1
    valid_values = flattened_tensor * valid_mask.float()

    valid_min = valid_values.masked_fill(~valid_mask, float('inf')).min(dim=1, keepdim=True)[0]
    valid_max = valid_values.masked_fill(~valid_mask, float('-inf')).max(dim=1, keepdim=True)[0]

    scale = (valid_max - valid_min).clamp(min=1e-7)

    normalized_values = (valid_values - valid_min) / scale

    scaled_values = normalized_values * (max_val - min_val) + min_val

    result_tensor.view(-1, tensor.shape[dim])[valid_mask] = scaled_values[valid_mask]

    return result_tensor

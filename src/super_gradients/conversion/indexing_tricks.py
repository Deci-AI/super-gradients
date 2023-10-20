import torch
from torch import Tensor


def gather_with_boolean_mask(data: Tensor, mask: Tensor, data_length: int) -> Tensor:
    """
    Workaround method for data[mask] with boolean mask for TensorRT 8.4.2.
    TensorRT does not support boolean masks, so we have to use TopK to get the indices of the True values and then
    use gather to get the values at those indices.
    Lastly we have to slice the result to only include the True values.
    The result is the same as data[mask] and the only downside is that TopK in TensorRT requires a fixed length
    k parameter, so we have to pass in the length of the data's first dimension.

    :param data:        N-D tensor
    :param mask:        Single dimension boolean mask
    :param data_length: Fixed length of the data's first dimension (TopK requires fixed length)
    :return:            Result of data[mask]
    """
    if data.size(0) != data_length:
        raise ValueError(f"Data must have the same first dimension as data_length. Got {data.size(0)} and {data_length}")
    if data.size(0) != mask.size(0):
        raise ValueError(f"Data and mask must have the same first dimension. Got {data.size(0)} and {mask.size(0)}")
    if mask.ndim != 1:
        raise ValueError(f"Mask must be single dimension. Got {mask.ndim}")

    mask = mask.float()
    topk_results = torch.topk(mask, k=data_length, dim=0, largest=True, sorted=True)
    data = torch.index_select(data, dim=0, index=topk_results.indices)

    # Now data is sorted by mask so that True values are at the top
    num_elements = torch.sum(mask).long()
    return data[0:num_elements]  # Return only the True values


def gather_with_float_mask(data: Tensor, mask: Tensor, data_length: int) -> Tensor:
    """
    A special version of gather_with_boolean_mask that works with float masks.
    A float mask incorporates the confidence of each item in the data and boolean mask itself.
    The way it is obtained is multiplication: scores * mask

    So when doing TopK we guarantee the order of the items is preserved w.r.t to scores and
    later we slice the result to only include elements that have a mask > 0.

    :param data:        N-D tensor
    :param mask:        Single dimension boolean mask
    :param data_length: Fixed length of the data's first dimension (TopK requires fixed length)
    :return:            Result of data[mask]
    """
    if data.size(0) != data_length:
        raise ValueError(f"Data must have the same first dimension as data_length. Got {data.size(0)} and {data_length}")
    if data.size(0) != mask.size(0):
        raise ValueError(f"Data and mask must have the same first dimension. Got {data.size(0)} and {mask.size(0)}")
    if mask.ndim != 1:
        raise ValueError(f"Mask must be single dimension. Got {mask.ndim}")

    topk_results = torch.topk(mask.float(), k=data_length, dim=0, largest=True, sorted=True)
    data = torch.index_select(data, dim=0, index=topk_results.indices)

    # Now data is sorted by mask so that True values are at the top
    num_elements = torch.sum(mask > 0).long()
    return data[0:num_elements]  # Return only the True values

import torch


# Based on https://github.com/odegeasslbc/Differentiable-RGB-to-HSV-convertion-pytorch/blob/master/pytorch_hsv.py


def rgb_decompose_median(rgb_tensors: torch.Tensor) -> tuple:
    """
    Separate median color from given RGB images.
    :param rgb_tensors: RGB images [batch_size, 3, height, width].
    :return: tuple(Delta RGB image [batch_size, 3, height, width], median R [batch_size], median G [batch_size], median B [batch_size])
    """
    assert len(rgb_tensors.shape) == 4 and rgb_tensors.shape[1] == 3

    _median_r = rgb_tensors[:, 0, :, :].view(rgb_tensors.shape[0], -1).median(dim=1)[0].detach()
    median_r = _median_r.unsqueeze(1).unsqueeze(2).expand(
        [rgb_tensors.shape[0], rgb_tensors.shape[2], rgb_tensors.shape[3]])

    _median_g = rgb_tensors[:, 1, :, :].view(rgb_tensors.shape[0], -1).median(dim=1)[0].detach()
    median_g = _median_g.unsqueeze(1).unsqueeze(2).expand(
        [rgb_tensors.shape[0], rgb_tensors.shape[2], rgb_tensors.shape[3]])

    _median_b = rgb_tensors[:, 2, :, :].view(rgb_tensors.shape[0], -1).median(dim=1)[0].detach()
    median_b = _median_b.unsqueeze(1).unsqueeze(2).expand(
        [rgb_tensors.shape[0], rgb_tensors.shape[2], rgb_tensors.shape[3]])

    updated_rgb_tensors = rgb_tensors.clone()
    updated_rgb_tensors[:, 0, :, :] -= median_r
    updated_rgb_tensors[:, 1, :, :] -= median_g
    updated_rgb_tensors[:, 2, :, :] -= median_b

    return updated_rgb_tensors, _median_r, _median_g, _median_b


def rgb_recombine_median(d_rgb_tensors: torch.Tensor, median_r: torch.Tensor, median_g: torch.Tensor, median_b: torch.Tensor) -> torch.Tensor:
    """
    Recombine separated median RGB to the delta RGB tensor.
    :param d_rgb_tensors: Delta RGB tensor [batch_size, 3, height, width].
    :param median_r: Median R [batch_size]
    :param median_g: Median G [batch_size]
    :param median_b: Median B [batch_size]
    :return: Recombined RGB texture [batch_size, 3, height, width]. Outputs in the range 0..1
    """
    assert len(d_rgb_tensors.shape) == 4 and d_rgb_tensors.shape[1] == 3
    assert len(median_r.shape) == 1
    assert len(median_g.shape) == 1
    assert len(median_b.shape) == 1

    updated_rgb_tensors = d_rgb_tensors.clone()
    median_r = median_r.unsqueeze(1).unsqueeze(2).expand(
        [updated_rgb_tensors.shape[0], updated_rgb_tensors.shape[2], updated_rgb_tensors.shape[3]])
    median_g = median_g.unsqueeze(1).unsqueeze(2).expand(
        [updated_rgb_tensors.shape[0], updated_rgb_tensors.shape[2], updated_rgb_tensors.shape[3]])
    median_b = median_b.unsqueeze(1).unsqueeze(2).expand(
        [updated_rgb_tensors.shape[0], updated_rgb_tensors.shape[2], updated_rgb_tensors.shape[3]])

    updated_rgb_tensors[:, 0, :, :] += median_r
    updated_rgb_tensors[:, 1, :, :] += median_g
    updated_rgb_tensors[:, 2, :, :] += median_b

    updated_rgb_tensors = updated_rgb_tensors.clamp(0, 1)

    return updated_rgb_tensors


def hsv_decompose_median(hsv_tensors: torch.Tensor):
    """
    Separate median color from given HSV images.
    :param hsv_tensors: HSV images [batch_size, 3, height, width]
    :return: tuple(Delta HSV image [batch_size, 3, height, width], median H [batch_size], median S [batch_size], median V [batch_size]).
            Output values are approximately in the range -0.5 to 0.5
    """
    assert len(hsv_tensors.shape) == 4 and hsv_tensors.shape[1] == 3

    # Compute the median color in RGB. HSV is tricky due to circular nature of hue.
    rgb_tensors = hsv_to_rgb(hsv_tensors)
    _median_r = rgb_tensors[:, 0, :, :].view(hsv_tensors.shape[0], -1).median(dim=1)[0].detach()
    _median_g = rgb_tensors[:, 1, :, :].view(hsv_tensors.shape[0], -1).median(dim=1)[0].detach()
    _median_b = rgb_tensors[:, 2, :, :].view(hsv_tensors.shape[0], -1).median(dim=1)[0].detach()

    # Convert median color to HSV.
    _median_hsv = rgb_to_hsv(torch.cat(
        [_median_r.view(-1, 1).unsqueeze(2).unsqueeze(3), _median_g.view(-1, 1).unsqueeze(2).unsqueeze(3),
         _median_b.view(-1, 1).unsqueeze(2).unsqueeze(3)], dim=1)).squeeze(3).squeeze(2)

    _median_h = _median_hsv[:, 0]
    _median_s = _median_hsv[:, 1]
    _median_v = _median_hsv[:, 2]

    median_h = _median_h.unsqueeze(1).unsqueeze(2).expand(
        [hsv_tensors.shape[0], hsv_tensors.shape[2], hsv_tensors.shape[3]])
    median_s = _median_s.unsqueeze(1).unsqueeze(2).expand(
        [hsv_tensors.shape[0], hsv_tensors.shape[2], hsv_tensors.shape[3]])
    median_v = _median_v.unsqueeze(1).unsqueeze(2).expand(
        [hsv_tensors.shape[0], hsv_tensors.shape[2], hsv_tensors.shape[3]])

    updated_hsv_tensors = hsv_tensors.clone()
    updated_hsv_tensors[:, 0, :, :] -= median_h
    updated_hsv_tensors[:, 0, :, :] = (updated_hsv_tensors[:, 0, :, :] + 1.5) % 1.0 - 0.5

    updated_hsv_tensors[:, 1, :, :] -= median_s
    updated_hsv_tensors[:, 2, :, :] -= median_v

    return updated_hsv_tensors, _median_h, _median_s, _median_v


def hsv_recombine_median(d_hsv_tensors, median_h, median_s, median_v):
    """
    Recombine separated median HSV to the delta HSV tensor.
    :param d_hsv_tensors: Delta HSV tensor [batch_size, 3, height, width].
    :param median_h: Median H [batch_size]
    :param median_s: Median S [batch_size]
    :param median_v: Median V [batch_size]
    :return: Recombined HSV texture [batch_size, 3, height, width]. Outputs in the range 0..1
    """
    assert len(d_hsv_tensors.shape) == 4 and d_hsv_tensors.shape[1] == 3
    assert len(median_h.shape) == 1
    assert len(median_s.shape) == 1
    assert len(median_v.shape) == 1

    updated_hsv_tensors = d_hsv_tensors.clone()

    median_h = median_h.unsqueeze(1).unsqueeze(2).expand(
        [updated_hsv_tensors.shape[0], updated_hsv_tensors.shape[2], updated_hsv_tensors.shape[3]])
    median_s = median_s.unsqueeze(1).unsqueeze(2).expand(
        [updated_hsv_tensors.shape[0], updated_hsv_tensors.shape[2], updated_hsv_tensors.shape[3]])
    median_v = median_v.unsqueeze(1).unsqueeze(2).expand(
        [updated_hsv_tensors.shape[0], updated_hsv_tensors.shape[2], updated_hsv_tensors.shape[3]])

    updated_hsv_tensors[:, 0, :, :] += 0.5
    updated_hsv_tensors[:, 0, :, :] += median_h
    updated_hsv_tensors[:, 0, :, :] = (updated_hsv_tensors[:, 0, :, :] + 1.5) % 1.0

    updated_hsv_tensors[:, 1, :, :] += median_s
    updated_hsv_tensors[:, 2, :, :] += median_v

    updated_hsv_tensors = updated_hsv_tensors.clamp(0, 1)

    return updated_hsv_tensors


def rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    """
    RGB to HSV conversion function.
    :param img: Batch of RGB images [batch_size, 3, height, width]
    :return: Batch of HSV images [batch_size, 3, height, width]
    """
    assert len(img.shape) == 4 and img.shape[1] == 3
    eps = 1e-7
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 2] == img.max(1)[0]]
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 1] == img.max(1)[0]]
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 0] == img.max(1)[0]]) % 6

    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue = hue / 6

    saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
    saturation[img.max(1)[0] == 0] = 0

    value = img.max(1)[0]

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value], dim=1)
    return hsv


def hsv_to_rgb(input_hsv_tensor):
    """
    Differentiable HSV to RGB conversion function.
    :param input_hsv_tensor: Batch of HSV images [batch_size, 3, height, width]
    :return: Batch of RGB images [batch_size, 3, height, width]
    """
    assert len(input_hsv_tensor.shape) == 4 and input_hsv_tensor.shape[1] == 3
    hues = input_hsv_tensor[:, 0, :, :]
    sats = input_hsv_tensor[:, 1, :, :]
    vals = input_hsv_tensor[:, 2, :, :]
    c = sats * vals

    x = c * (1 - torch.abs((hues * 6.0) % 2.0 - 1.0))
    m = vals - c

    # Compute R
    r_hat = torch.zeros_like(hues)
    filter_hues = hues.clone()
    r_hat[filter_hues < 1.0 / 6.0] = c[filter_hues < 1.0 / 6.0]
    filter_hues[filter_hues < 1.0 / 6.0] += 10.0
    r_hat[filter_hues < 2.0 / 6.0] = x[filter_hues < 2.0 / 6.0]
    filter_hues[filter_hues < 2.0 / 6.0] += 10.0
    r_hat[filter_hues < 3.0 / 6.0] = 0
    filter_hues[filter_hues < 3.0 / 6.0] += 10.0
    r_hat[filter_hues < 4.0 / 6.0] = 0
    filter_hues[filter_hues < 4.0 / 6.0] += 10.0
    r_hat[filter_hues < 5.0 / 6.0] = x[filter_hues < 5.0 / 6.0]
    filter_hues[filter_hues < 5.0 / 6.0] += 10.0
    r_hat[filter_hues <= 6.0 / 6.0] = c[filter_hues <= 6.0 / 6.0]
    filter_hues[filter_hues <= 6.0 / 6.0] += 10.0

    # Compute G
    g_hat = torch.zeros_like(hues)
    filter_hues = hues.clone()
    g_hat[filter_hues < 1.0 / 6.0] = x[filter_hues < 1.0 / 6.0]
    filter_hues[filter_hues < 1.0 / 6.0] += 10.0
    g_hat[filter_hues < 2.0 / 6.0] = c[filter_hues < 2.0 / 6.0]
    filter_hues[filter_hues < 2.0 / 6.0] += 10.0
    g_hat[filter_hues < 3.0 / 6.0] = c[filter_hues < 3.0 / 6.0]
    filter_hues[filter_hues < 3.0 / 6.0] += 10.0
    g_hat[filter_hues < 4.0 / 6.0] = x[filter_hues < 4.0 / 6.0]
    filter_hues[filter_hues < 4.0 / 6.0] += 10.0
    g_hat[filter_hues < 5.0 / 6.0] = 0
    filter_hues[filter_hues < 5.0 / 6.0] += 10.0
    g_hat[filter_hues <= 6.0 / 6.0] = 0
    filter_hues[filter_hues <= 6.0 / 6.0] += 10.0

    # Compute B
    b_hat = torch.zeros_like(hues)
    filter_hues = hues.clone()
    b_hat[filter_hues < 1.0 / 6.0] = 0
    filter_hues[filter_hues < 1.0 / 6.0] += 10.0
    b_hat[filter_hues < 2.0 / 6.0] = 0
    filter_hues[filter_hues < 2.0 / 6.0] += 10.0
    b_hat[filter_hues < 3.0 / 6.0] = x[filter_hues < 3.0 / 6.0]
    filter_hues[filter_hues < 3.0 / 6.0] += 10.0
    b_hat[filter_hues < 4.0 / 6.0] = c[filter_hues < 4.0 / 6.0]
    filter_hues[filter_hues < 4.0 / 6.0] += 10.0
    b_hat[filter_hues < 5.0 / 6.0] = c[filter_hues < 5.0 / 6.0]
    filter_hues[filter_hues < 5.0 / 6.0] += 10.0
    b_hat[filter_hues <= 6.0 / 6.0] = x[filter_hues <= 6.0 / 6.0]
    filter_hues[filter_hues <= 6.0 / 6.0] += 10.0

    r = (r_hat + m).view(input_hsv_tensor.shape[0], 1, input_hsv_tensor.shape[2],
                         input_hsv_tensor.shape[3])
    g = (g_hat + m).view(input_hsv_tensor.shape[0], 1, input_hsv_tensor.shape[2],
                         input_hsv_tensor.shape[3])
    b = (b_hat + m).view(input_hsv_tensor.shape[0], 1, input_hsv_tensor.shape[2],
                         input_hsv_tensor.shape[3])

    rgb = torch.cat([r, g, b], dim=1)
    return rgb

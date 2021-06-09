import torchvision
import torch

def get_medoid_key(textures: dict) -> str:
    """
    Given a dictionary of keys and images, return the key of the medoid image.
    :param textures: Dictionary of keys and images
    :return: Medoid key
    """
    texture_vectors = []
    key_list = []
    for crop_key, crop_image in textures.items():
        key_list.append(crop_key)
        texture_vectors.append(torchvision.transforms.ToTensor()(crop_image.image).view(1, -1))
    texture_vectors = torch.cat(texture_vectors, dim=0)
    mean_vector = texture_vectors.mean(dim=0)
    delta_vectors = texture_vectors - mean_vector.unsqueeze(0).repeat(texture_vectors.shape[0], 1)
    distances = (delta_vectors ** 2).sum(dim=1) ** 0.5
    min_distance, min_index = distances.min(dim=0)
    return key_list[min_index]

from plan2scene.config_manager import ConfigManager
from plan2scene.common.residence import House, Room
import torch


def get_house_graph(conf: ConfigManager, house: House, surface_maskout_map: dict, key="prop"):
    """
    Generates node embeddings and edge pairs for a given house.
    :param conf: ConfigManager
    :param house: House to generate graph
    :param surface_maskout_map: Dictionary of surfaces to be dropped from input. {room_index: [list of surfaces. e.g. 'floor', 'wall', 'ceiling']}
    :return: Pair of node embeddings tensor and edge indices tensor
    """
    combined_emb_dim = conf.texture_gen.combined_emb_dim
    node_embeddings = []
    surface_embeddings = []
    for room_index, room in house.rooms.items():
        assert isinstance(room, Room)
        # Room Type
        node_embedding = []
        rt_embedding = torch.zeros(len(conf.room_types))
        for rt in room.types:
            rt_embedding[conf.room_types.index(rt)] = 1.0
        node_embedding.append(rt_embedding.view(1, -1))

        short_embeddings = []
        for surf in conf.surfaces:
            if room_index in surface_maskout_map and surf in surface_maskout_map[room_index]: # Masked out
                short_surf_embedding = torch.zeros((combined_emb_dim,)).view(1, -1)
                surf_present = False
            elif key not in room.surface_embeddings[surf]: # Unobserved
                short_surf_embedding = torch.zeros((combined_emb_dim,)).view(1, -1)
                surf_present = False
            else:
                short_surf_embedding = room.surface_embeddings[surf][key].detach()
                surf_present = True

            surf_embedding = torch.cat([torch.tensor([[surf_present]], dtype=torch.float32), short_surf_embedding],
                                        dim=1)
            node_embedding.append(surf_embedding)
            short_embeddings.append(short_surf_embedding)
            del surf_embedding
            del surf_present
            del short_surf_embedding

        if conf.texture_prop.graph_generator.include_enable_in_target:
            assert False

        surface_embeddings.append(
            torch.cat(short_embeddings, dim=0).unsqueeze(0))
        node_embedding_tensor = torch.cat(node_embedding, dim=1)
        node_embeddings.append(node_embedding_tensor)

    node_embeddings_tensor = torch.cat(node_embeddings, dim=0)
    surface_embeddings_tensor = torch.cat(surface_embeddings, dim=0)

    edge_indices = []
    for r1_index, r2_index in house.door_connected_room_pairs:
        if r2_index >= 0:
            edge_indices.append([r1_index, r2_index])
    edge_indices_tensor = torch.tensor(edge_indices, dtype=torch.long)

    return node_embeddings_tensor, edge_indices_tensor, surface_embeddings_tensor


def generate_target_and_mask(conf: ConfigManager, house: House, target_surface_map: dict, include_target: bool, key="prop") -> tuple:
    """
    Generates y and y_mask tensors for a given house, targetting surfaces that are indicated.
    :param conf: Config Manager
    :param house: House
    :param target_surface_map: Dictionary of room surfaces to include in mask and target. {room_index: [list of surfaces. e.g. 'floor', 'wall', 'ceiling']}
    :param include_target: Pass true to populate target embeddings of each node.
    :return: Pair of target tensor [node_count, surface_count, emb] and masks tensor [node_count, surface_count].
    """
    combined_emb_dim = conf.texture_gen.combined_emb_dim
    updated_combined_emb_dim = combined_emb_dim
    if conf.texture_prop.graph_generator.include_enable_in_target:
        updated_combined_emb_dim += 1

    if include_target:
        room_targets = []
    else:
        room_targets = None
    room_masks = []
    for room_index, room in house.rooms.items():
        if room_index not in target_surface_map:
            # Unlisted room
            if include_target:
                room_target = torch.zeros([1, len(conf.surfaces), updated_combined_emb_dim], dtype=torch.float)
                room_targets.append(room_target)
            room_masks.append(torch.zeros([1, len(conf.surfaces)], dtype=torch.bool))
            continue

        if include_target:
            room_target = []  # surface_count * [1, 1, combined_dim]

        room_mask = []  # surface_count

        for surf in conf.surfaces:
            if room_index in target_surface_map and surf in target_surface_map[room_index]:
                if include_target:
                    surf_target = room.surface_embeddings[surf][key].detach().unsqueeze(0)
                surf_mask = True
            else:
                if include_target:
                    surf_target = torch.zeros([1, 1, combined_emb_dim], dtype=torch.float)
                surf_mask = False

            if include_target:
                if conf.texture_prop.graph_generator.include_enable_in_target:
                    surf_target = torch.cat([torch.tensor([[[surf_mask]]], dtype=torch.float32), surf_target], dim=2)
                room_target.append(surf_target)
            room_mask.append(surf_mask)


        if include_target:
            room_target_tensor = torch.cat(room_target, dim=1)
            room_targets.append(room_target_tensor)

        room_mask_tensor = torch.tensor(room_mask, dtype=torch.bool).unsqueeze(0)
        room_masks.append(room_mask_tensor)

    if include_target:
        room_targets_tensor = torch.cat(room_targets, dim=0)
    else:
        room_targets_tensor = None

    room_masks_tensor = torch.cat(room_masks, dim=0)
    return room_targets_tensor, room_masks_tensor
# world/world_gen.py
import torch

def generate_random_world(x_size=200, y_size=200, z_size=50, device="cpu",
                          num_obstacles=8,
                          min_box_size=(5, 5, 5),
                          max_box_size=(40, 40, 30)):
    """
    Returns occupancy[x,y,z] as a torch boolean tensor.
    Obstacles: random boxes touching the ground.
    """

    occ = torch.zeros((x_size, y_size, z_size), dtype=torch.bool, device=device)

    for _ in range(num_obstacles):
        # Random width/length/height
        dx = torch.randint(min_box_size[0], max_box_size[0], (1,)).item()
        dy = torch.randint(min_box_size[1], max_box_size[1], (1,)).item()
        dz = torch.randint(min_box_size[2], max_box_size[2], (1,)).item()

        # Random XY position, Z is always ground-contact
        x = torch.randint(0, x_size - dx, (1,)).item()
        y = torch.randint(0, y_size - dy, (1,)).item()

        occ[x:x+dx, y:y+dy, 0:dz] = True

    return occ

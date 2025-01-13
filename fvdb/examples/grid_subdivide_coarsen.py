# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import uuid

import numpy as np
import point_cloud_utils as pcu
import polyscope as ps
import torch

import fvdb
from fvdb.utils.examples import load_dragon_mesh


def visualize_grid(a: fvdb.GridBatch, offset: float):
    assert a.grid_count == 1
    mesh_a = pcu.voxel_grid_geometry(a.ijk[0].jdata.cpu().numpy(), a.voxel_sizes[0].cpu().numpy())
    ps.register_surface_mesh(
        str(uuid.uuid4()),
        mesh_a[0] + np.array([0.0, 0.0, offset]) - a.voxel_sizes[0].cpu().numpy()[None, :] / 2.0,
        mesh_a[1],
        enabled=True,
    )


if __name__ == "__main__":
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_navigation_style("free")

    [p] = load_dragon_mesh(mode="v", device=torch.device("cuda"))


    grid_origin = fvdb.gridbatch_from_points(p, voxel_sizes=[0.2] * 3, origins=[0.0] * 3)
    visualize_grid(grid_origin, 0.0)
    ps.screenshot("outputs/grid_subdivide_coarsen_origin.png")
    print("-" * 80)
    print("grid_origin.ijk[0].jdata.shape: ", grid_origin.ijk[0].jdata.shape)
    print("grid_origin.num_enabled_voxels: ", grid_origin.num_enabled_voxels[0])

    grid_subdivided = grid_origin.subdivided_grid(2)
    visualize_grid(grid_subdivided, 0.15)
    ps.screenshot("outputs/grid_subdivide_coarsen_subdivided.png")
    print("-" * 80)
    print("grid_subdivided.ijk[0].jdata.shape: ", grid_subdivided.ijk[0].jdata.shape)
    print("grid_subdivided.num_enabled_voxels: ", grid_subdivided.num_enabled_voxels[0])


    grid_coarsened = grid_origin.coarsened_grid(2)
    visualize_grid(grid_coarsened, 0.3)
    ps.screenshot("outputs/grid_subdivide_coarsen_coarsened.png")
    print("-" * 80)
    print("grid_coarsened.ijk[0].jdata.shape: ", grid_coarsened.ijk[0].jdata.shape)
    print("grid_coarsened.num_enabled_voxels: ", grid_coarsened.num_enabled_voxels[0])

    grid_dual = grid_origin.dual_grid()

    grid_dual_gv, grid_dual_ge = grid_dual.viz_edge_network
    ps.remove_all_structures()
    visualize_grid(grid_origin, 0.0)
    ps.screenshot("outputs/grid_subdivide_coarsen_origin_dual.png")

    ps.register_curve_network(
        str(uuid.uuid4()),
        grid_dual_gv[0].jdata.cpu().numpy(),
        grid_dual_ge[0].jdata.cpu().numpy(),
        enabled=True,
        radius=0.004,
    )
    # ps.show()
    ps.screenshot("outputs/grid_subdivide_coarsen_origin_dual_edges.png")

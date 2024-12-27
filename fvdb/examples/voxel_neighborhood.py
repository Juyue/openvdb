# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import polyscope as ps
import torch

from fvdb import GridBatch, gridbatch_from_ijk
from fvdb.utils.examples import load_dragon_mesh


def main():
    device = "cuda"

    # vox_size = 0.0075
    vox_size = 0.1
    vox_origin = (0, 0, 0)
    N = 10

    [p] = load_dragon_mesh(mode="v", skip_every=N, device=torch.device(device))
    print("Total number of points sampled from mesh: ", p.shape[0])

    index = GridBatch(device=device)
    index.set_from_points(p, [-1, -1, -1], [1, 1, 1], vox_size, vox_origin)

    primal_voxels = index.ijk.jdata # (b, 3)
    nhood = index.neighbor_indexes(primal_voxels, 2, 0).jdata # (b, 5, 5, 5), index of neighbors, if [-1, -1, -1] then not a neighbor

    ps.init()
    for ii in range(10):
        randvox = np.random.randint(nhood.shape[0])

        voxijk = primal_voxels[randvox]
        nbrs = primal_voxels[nhood[randvox][nhood[randvox] >= 0]]
        nhood_ijk = torch.cat([voxijk.unsqueeze(0), nbrs], dim=0)

        vp, ve = index.viz_edge_network
        vp, ve = vp.jdata, ve.jdata

        vi, vei = gridbatch_from_ijk(nhood_ijk, voxel_sizes=vox_size, origins=vox_origin).viz_edge_network
        vi, vei = vi.jdata, vei.jdata

        ps.register_curve_network("vox", vp.cpu().numpy(), ve.cpu().numpy(), radius=0.0025)
        ps.register_curve_network("nhd", vi.cpu().numpy(), vei.cpu().numpy(), radius=0.005)
        ps.screenshot(f"outputs/voxel_neighborhood_{ii}.png")
        # ps.show()


if __name__ == "__main__":
    main()

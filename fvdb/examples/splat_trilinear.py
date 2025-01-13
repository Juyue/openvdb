# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import timeit
print("fvdb")

import polyscope as ps
import torch

from fvdb import GridBatch
from fvdb.utils.examples import load_dragon_mesh


def main():
    logging.basicConfig(level=logging.INFO)
    logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))

    device = torch.device("cuda")
    dtype = torch.float32

    vox_size = 0.0025
    vox_origin = (0, 0, 0)

    points, normals = load_dragon_mesh(skip_every=1, device=device, dtype=dtype)

    index = GridBatch(device=device)
    index.set_from_points(points, voxel_sizes=vox_size, origins=vox_origin)

    logging.info("Splatting into grid...")
    start = timeit.default_timer()
    nsplat = index.splat_trilinear(points, normals)
    if points.is_cuda:
        torch.cuda.synchronize()
    logging.info(f"Done in {timeit.default_timer() - start}s!")

    # grid_batch.ijk is a JaggedTensor of shape (N, 3), containing the voxel coordinates of the points
    gp = index.ijk
    # grid_batch.grid_to_world is a JaggedTensor of shape (N, 3), containing the world coordinates of the points
    gp = index.grid_to_world(gp.type(dtype))

    points, normals = points.cpu(), normals.cpu()
    nsplat = nsplat.cpu()
    gp = gp.cpu()

    ps.init()
    # ps.register_point_cloud("points", points, radius=0.00075)
    grid_pts = ps.register_point_cloud("vox coords", gp.jdata, radius=0.0005)

    grid_pts.add_vector_quantity("splatted normals", nsplat.jdata, enabled=True, length=0.05, radius=0.001)
    # ps.show()
    ps.screenshot("outputs/splat_trilinear.png");
    import pdb; pdb.set_trace()

    # index_dual = index.dual_grid()
    # gd = index_dual.ijk
    # gd = index_dual.grid_to_world(gd.type(dtype))
    # gd = gd.cpu()
    # ps.register_point_cloud("vox coords dual", gd.jdata, radius=0.0005)
    # ps.screenshot("splat_trilinear_dual.png");


if __name__ == "__main__":
    main()
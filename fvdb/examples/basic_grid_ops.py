import fvdb
import click
import torch
import numpy as np
import point_cloud_utils as pcu

import polyscope as ps
from fvdb.utils.examples import load_car_1_mesh, load_car_2_mesh

fvdb.nn.SparseConv3d.allow_tf32 = False
VOXEL_SIZES = 0.03
def visualize_grid_with_voxel_colors(grid, feature, fname="grid"):
    grid_mesh = pcu.voxel_grid_geometry(grid.ijk[0].jdata.detach().cpu().numpy(), grid.voxel_sizes[0].detach().cpu().numpy(), gap_fraction=0.1)
    grid_color = feature[0].jdata.detach().cpu().numpy().repeat(8, axis=0).reshape(-1, 3)
    ps.register_surface_mesh("grid", grid_mesh[0], grid_mesh[1], enabled=True).add_color_quantity("color", grid_color, enabled=True)
    ps.screenshot(f"outputs/{fname}.png")

def _create_grid_two_cars(voxel_sizes=VOXEL_SIZES):
    v1, f1 = load_car_1_mesh(mode="vf")
    v2, f2 = load_car_2_mesh(mode="vf")
    f1, f2 = f1.to(torch.int32), f2.to(torch.int32)

    mesh_v_jagged = fvdb.JaggedTensor([v1, v2]).cuda()
    mesh_f_jagged = fvdb.JaggedTensor([f1, f2]).cuda()
    grid = fvdb.gridbatch_from_mesh(mesh_v_jagged, mesh_f_jagged, voxel_sizes=voxel_sizes)

    # add corresponding feature. use coordinates as feature.
    feature = grid.grid_to_world(grid.ijk.float())
    feature_data = feature.jdata
    feature.jdata = torch.abs(feature_data)/ torch.norm(feature_data, dim=-1, keepdim=True)
    return grid, feature

def example_sampling_grids():
    """given world coordinates of a point, and a grid + corresponding feature, sample the feature at the point"""
    # 0. Prep
    N = 2
    grid, feature = _create_grid_two_cars()

    # 1. Trilinear sampling
    # 1.0 Sample a active voxel in the grid
    randvox_idx = torch.tensor(np.random.randint(low=grid.total_voxels, size=N))
    # 1.1 Get the center of the voxel
    randvox_ijk = grid.ijk.jdata[randvox_idx].cuda() # This needs to be a JaggedTensor
    randvox_world = grid.grid_to_world(randvox_ijk.float().view(-1, 3)).jdata
    # 1.2 Add some noise to the voxel center
    sampled_point = randvox_world + torch.randn_like(randvox_world) * VOXEL_SIZES  * 0.3
    # 1.3 Sample the feature at the point
    sampled_feature = grid.sample_trilinear(sampled_point, feature)

    # 2. Visualize
    ps.init()
    # 2.0 Visualize the vertices and edges of the sampled voxel
    randvox_grid = fvdb.GridBatch().cuda()
    randvox_grid.set_from_ijk(randvox_ijk, voxel_sizes=VOXEL_SIZES)
    voxel_v_all, voxel_e_all = randvox_grid.viz_edge_network
    print("Sampled grid", voxel_v_all.jdata.shape, voxel_e_all.jdata.shape)
    ps.register_curve_network("voxel", voxel_v_all.jdata.cpu().numpy(), voxel_e_all.jdata.cpu().numpy())
    # 2.1 Visualize the sampled points, and the corresponding feature
    ps.register_point_cloud("sampled_points", sampled_point.cpu().numpy()).add_color_quantity("color", sampled_feature.jdata.cpu().numpy(), enabled=True)
    ps.screenshot("outputs/sampling_grids.png")

def example_splating_grids():
    """given a grid, and a point cloud, sample the grid at the point cloud"""

    #0. Prep:randomly create a point, and a color.
    N = 4
    points = torch.randn(N, 3).cuda()
    grid = fvdb.GridBatch().cuda()
    grid.set_from_points(points, voxel_sizes=VOXEL_SIZES)

    colors = torch.randn(N, 3).cuda()
    colors = torch.abs(colors)/torch.norm(colors, dim=-1, keepdim=True)

    # 1. Splat the colors to the grid
    feature = grid.splat_trilinear(points, colors)

    # 2. Visualize
    ps.init()
    grid_mesh = pcu.voxel_grid_geometry(grid.ijk.jdata.cpu().numpy(), grid.voxel_sizes[0].cpu().numpy(), gap_fraction=0.1)
    grid_color = feature.jdata.cpu().numpy().repeat(8, axis=0).reshape(-1, 3) # every voxel has 8 vertices
    ps.register_surface_mesh("grid", grid_mesh[0], grid_mesh[1], enabled=True).add_color_quantity("color", grid_color, enabled=True)
    ps.screenshot("outputs/splating_grids_mesh.png")

    ps.remove_all_structures()
    ps.register_point_cloud("points", points.cpu().numpy()).add_color_quantity("color", colors.cpu().numpy(), enabled=True)
    ps.screenshot("outputs/splating_grids_points.png")

def example_checking_if_points_are_inside_grid():

    # 0. Prep
    N = 1
    points = torch.randn(N, 3).cuda()
    # 0.1 create a grid of 4 active voxels
    grid = fvdb.GridBatch().cuda()
    grid.set_from_points(points, voxel_sizes=VOXEL_SIZES)

    # 0.2 generate some random points in the bbox of each grid.
    M = 10
    sampled_points = points.view(N, -1, 3) + torch.randn((N, M, 3)).cuda() * VOXEL_SIZES * 0.3
    sampled_points = sampled_points.view(-1, 3)
    sampled_grid = fvdb.GridBatch().cuda()
    sampled_grid.set_from_points(sampled_points, voxel_sizes=VOXEL_SIZES)
    M2 = sampled_grid.total_voxels
    print("Number of voxels in sampled grid: ", M2)

    # 1. Check if the points are inside the grid
    points_inside = grid.points_in_active_voxel(sampled_points)
    grid_inside = grid.coords_in_active_voxel(sampled_grid.ijk) # JaggedTensor
    print("Number of points inside: ", points_inside.jdata.sum().item())
    print("Number of grid inside: ", grid_inside.jdata.sum().item())

    # 2. Visualize
    ps.init()
    # visualize the voxel grid;
    gv, ve = grid.viz_edge_network
    ps.register_curve_network("grid", gv.jdata.cpu().numpy(), ve.jdata.cpu().numpy(), enabled=True)

    # visualize the points
    colors_1 = np.ones((N * M, 3)) * 0.5
    colors_1[points_inside.jdata.cpu().numpy()] = np.array([1, 0, 0])
    ps.register_point_cloud("points_1", sampled_points.cpu().numpy()).add_color_quantity("color", colors_1, enabled=True)

    points_2 = sampled_grid.grid_to_world(sampled_grid.ijk.float())
    colors_2 = np.ones((M2, 3)) * 0.1
    colors_2[grid_inside.jdata.cpu().numpy()] = np.array([0, 1, 0])
    ps.register_point_cloud("points_2", points_2.jdata.cpu().numpy()).add_color_quantity("color", colors_2, enabled=True)
    ps.screenshot("outputs/checking_if_points_are_inside_grid.png")

def example_indexing_grids():
    """
    reference_grid: GridBatch
    query_ijk (query_grid): TorchTensor/JaggedTensor

    idx_into_reference: JaggedTensor
    idx_into_query: JaggedTensor
    """
    # 0. Prep
    N = 10
    points = torch.randn(N, 3).cuda()
    print("points", points.shape)
    # 0.1 create a reference grid
    reference_grid = fvdb.GridBatch().cuda()
    reference_grid.set_from_points(points, voxel_sizes=VOXEL_SIZES)
    # 0.2 create a query ijk
    M = 2
    ijk = reference_grid.ijk.jdata
    randperm_idx = torch.randperm(ijk.shape[0])[:M]
    query_ijk = ijk[randperm_idx]
    # 0.3 create a query grid
    query_grid = fvdb.GridBatch().cuda()
    query_grid.set_from_ijk(query_ijk, voxel_sizes=VOXEL_SIZES)
    print("-"*100)
    print("query_ijk\n", query_ijk)
    print("query_grid.ijk\n", query_grid.ijk.jdata)

    # 1. Get the idx of query_ijk in reference_grid
    idx_into_reference = reference_grid.ijk_to_index(query_ijk) # query_grid works too
    print("-"*100)
    print("idx_into_reference\n", idx_into_reference.jdata)
    print("reference_grid.ijk[idx_into_reference]\n", reference_grid.ijk[idx_into_reference].jdata)

    # 2. Get the inverse idx.
    idx_into_query_raw = reference_grid.ijk_to_inv_index(query_ijk)
    idx_into_query = idx_into_query_raw.rmask(idx_into_query_raw.jdata >= 0)

    print("-"*100)
    print("idx_into_query_raw\n", idx_into_query_raw.jdata)
    print("idx_into_query\n", idx_into_query.jdata)
    print("WRONG: query_grid.ijk[idx_into_query_raw]\n", query_grid.ijk[idx_into_query_raw].jdata)
    print("CORRECT: query_grid.ijk[idx_into_query.rmask]\n", query_grid.ijk[idx_into_query].jdata)

def example_neighboring_voxels():
    """given a grid, and a point, get the neighboring voxels of the point"""
    # 0. Prep
    N = 10
    points = torch.randn(N, 3).cuda()
    grid = fvdb.GridBatch().cuda()
    grid.set_from_points(points, voxel_sizes=VOXEL_SIZES)
    finer_grid = grid.subdivided_grid(4)

    # 1. Get the neighboring voxels
    M = 2
    rand_ijk = finer_grid.ijk.jdata[torch.randperm(finer_grid.total_voxels)[:M]]
    neighboring_idxs = finer_grid.neighbor_indexes(rand_ijk, 2).jdata.view(M, -1)
    neighboring_ijk = [None] * M
    for ii in range(M):
        idx = neighboring_idxs[ii]
        print(f"There are {idx[idx >= 0].shape[0]} neighboring voxels for point {ii}")
        neighboring_ijk[ii] = finer_grid.ijk.jdata[idx[idx >= 0]]
    neighboring_ijk = torch.cat(neighboring_ijk, dim=0)

    # 2. Visualize
    # 2.0 visualize the grid
    ps.init()
    finer_grid_v, finer_grid_e = finer_grid.viz_edge_network
    ps.register_curve_network("finer_grid", finer_grid_v.jdata.cpu().numpy(), finer_grid_e.jdata.cpu().numpy(), enabled=True)
    # 2.1 visualize the neighboring voxels
    # 2.1.1, create a voxel grid geometry of all neighboring voxels.
    neighboring_grid = fvdb.GridBatch().cuda()
    neighboring_grid.set_from_ijk(neighboring_ijk.view(-1, 3), voxel_sizes=finer_grid.voxel_sizes)

    rand_ijk_idx = neighboring_grid.ijk_to_index(rand_ijk)

    neighboring_colors = np.ones((neighboring_ijk.shape[0], 3)) * 0.5 # grey
    neighboring_colors[rand_ijk_idx.jdata.cpu().numpy()] = np.array([1, 0, 0])
    neighboring_colors = neighboring_colors.repeat(8, axis=0).reshape(-1, 3)
    voxel_sizes = finer_grid.voxel_sizes.cpu().numpy()
    neighboring_mesh = pcu.voxel_grid_geometry(neighboring_ijk.cpu().numpy(), voxel_sizes[0], gap_fraction=0.1)

    ps.register_surface_mesh("neighboring_voxels", neighboring_mesh[0], neighboring_mesh[1], enabled=True).add_color_quantity("color", neighboring_colors, enabled=True)
    ps.screenshot("outputs/neighboring_voxels.png")

def example_pooling_or_subdivide():
    """
    grid: GridBatch
    feature: JaggedTensor


    finer_grid: GridBatch
    finer_feature: JaggedTensor

    coarser_grid: GridBatch
    coarser_feature: JaggedTensor
    """

    # 0. Prep, load a grid of cars and corresponding normals
    grid, feature = _create_grid_two_cars()

    # 1. coarsen the grid
    coarse_feature, coarse_grid = grid.max_pool(2, feature)
    # 2. subdivide the grid
    fine_feature, fine_grid = grid.subdivide(2, feature )
    # 2.1 subdivide wo original grid,
    fine_feature_wo_original_grid, fine_grid_wo_original_grid = coarse_grid.subdivide(2, coarse_feature)
    # 2.2 subdivide with original grid,
    recovered_feature, recovered_grid = coarse_grid.subdivide(2, coarse_feature, fine_grid=grid)


    # 3. Visualize
    ps.init()
    # visualize the voxel mesh
    visualize_grid_with_voxel_colors(grid, feature, fname="original_grid")
    ps.remove_all_structures()
    visualize_grid_with_voxel_colors(coarse_grid, coarse_feature, fname="grid_after_max_pooling")
    ps.remove_all_structures()
    visualize_grid_with_voxel_colors(fine_grid, fine_feature, fname="grid_after_subdividing")
    ps.remove_all_structures()
    visualize_grid_with_voxel_colors(fine_grid_wo_original_grid, fine_feature_wo_original_grid, fname="grid_coarse_then_subdivide_wo_original_grid")
    ps.remove_all_structures()
    visualize_grid_with_voxel_colors(recovered_grid, recovered_feature, fname="grid_coarse_then_subdivide_keep_original_grid")

def example_num_voxels_in_grid():
    """given a grid, get the number of voxels in the grid"""

    # 0. Prep
    grid, _ = _create_grid_two_cars()

    for bb, nn in enumerate(grid.num_voxels):
        print(f"Number of voxels in grid {bb}: {nn}")

    for bb, nn in enumerate(grid.num_enabled_voxels):
        print(f"Number of enabled voxels in grid {bb}: {nn}")

def example_convolution():
    """given a grid, and a kernel, convolve the grid with the kernel"""

    # 0. Prep
    grid, feature = _create_grid_two_cars()
    import ipdb; ipdb.set_trace()

    # 1.
    vdbtensor = fvdb.nn.VDBTensor(grid, feature)
    conv = fvdb.nn.SparseConv3d(in_channels=3, out_channels=3, kernel_size=3, stride=1, bias=False).to(vdbtensor.device)

    output = conv(vdbtensor)

    conv_stride_2 = fvdb.nn.SparseConv3d(in_channels=3, out_channels=3, kernel_size=3, stride=2, bias=False).to(vdbtensor.device)
    output_stride_2 = conv_stride_2(vdbtensor)

    transposed_conv = fvdb.nn.SparseConv3d(in_channels=3, out_channels=3, kernel_size=3, stride=1, bias=False, transposed=True).to(vdbtensor.device)
    output_transposed = transposed_conv(output_stride_2, out_grid = grid)

    # visualize two tensors
    ps.init()
    visualize_grid_with_voxel_colors(grid, feature, fname="original_grid")
    visualize_grid_with_voxel_colors(output.grid, output.data, fname="convolution_output")
    visualize_grid_with_voxel_colors(output_stride_2.grid, output_stride_2.data, fname="convolution_output_stride_2")
    visualize_grid_with_voxel_colors(output_transposed.grid, output_transposed.data, fname="convolution_output_transposed")


@click.command()
@click.option("--mode", type=click.Choice(["00", "01", "02", "03", "04", "05", "06", "07"]), default="07")
def main(mode):
    if mode == "sampling_grids" or mode == "00":
        example_sampling_grids()
    elif mode == "splating_grids" or mode == "01":
        example_splating_grids()
    elif mode == "checking_if_points_are_inside_grid" or mode == "02":
        example_checking_if_points_are_inside_grid()
    elif mode == "indexing_grids" or mode == "03":
        example_indexing_grids()
    elif mode == "neighboring_voxels" or mode == "04":
        example_neighboring_voxels()
    elif mode == "pooling_or_subdivide" or mode == "05":
        example_pooling_or_subdivide()
    elif mode == "num_voxels_in_grid" or mode == "06":
        example_num_voxels_in_grid()
    elif mode == "convolution" or mode == "07":
        example_convolution()

if __name__ == "__main__":
    main()

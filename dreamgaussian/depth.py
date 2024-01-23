# Copyright (c) Microsoft Corporation.   
# Licensed under the MIT License.

# differentiable point cloud rendering
import torch
from p2i_op import p2i
import open3d
import numpy as np
def normalize(x, dim):
    return x / torch.max(x.norm(None, dim=dim, keepdim=True), torch.tensor(1e-6, dtype=x.dtype, device=x.device))


def look_at(eyes, centers, ups):
    """look at
    Inputs:
    - eyes: float, [batch x 3], position of the observer's eye
    - centers: float, [batch x 3], where the observer is looking at (the lookat point in the above image)
    - ups: float, [batch x 3], the upper head direction of the observer

    Returns:
    - view_mat: float, [batch x 4 x 4]
    """
    B = eyes.shape[0]

    # get the directions (unit vectors) for camera x, y, z axes in the world coordinate
    zaxis = normalize(eyes - centers, dim=1)  # The z-axis points from the object to the observer
    xaxis = normalize(torch.cross(ups, zaxis, dim=1), dim=1)  # get the x-axis(The 'right' direction of the observer)
    yaxis = torch.cross(zaxis, xaxis, dim=1)  # get the y-axis(The rectified 'up' direction of the observer)

    # constant 0 or 1 placeholders
    zeros_pl = torch.zeros([B], dtype=eyes.dtype, device=eyes.device)
    ones_pl = torch.ones([B], dtype=eyes.dtype, device=eyes.device)

    translation = torch.stack(
        [
            ones_pl,
            zeros_pl,
            zeros_pl,
            -eyes[:, 0],
            zeros_pl,
            ones_pl,
            zeros_pl,
            -eyes[:, 1],
            zeros_pl,
            zeros_pl,
            ones_pl,
            -eyes[:, 2],
            zeros_pl,
            zeros_pl,
            zeros_pl,
            ones_pl,
        ],
        -1,
    ).view(
        -1, 4, 4
    )  # translate coordinates so that the eyes becomes (0,0,0)

    orientation = torch.stack(
        [
            xaxis[:, 0],
            xaxis[:, 1],
            xaxis[:, 2],
            zeros_pl,
            yaxis[:, 0],
            yaxis[:, 1],
            yaxis[:, 2],
            zeros_pl,
            zaxis[:, 0],
            zaxis[:, 1],
            zaxis[:, 2],
            zeros_pl,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            ones_pl,
        ],
        -1,
    ).view(
        -1, 4, 4
    )  # rotate the coordinates so that the above zaxis becomes (0,0,1), yaxis becomes (0,1,0), xaxis becomes (1,0,0)

    return orientation @ translation  # first translate, then orientate


def perspective(fovy, aspect, z_near, z_far):
    """perspective (right hand_no)
    Simulates the way the human eye perceives the world, where objects farther away appear smaller. The matrix converges lines that are parallel in the world space to meet at a vanishing point in the image, creating a sense of depth.
    Inputs:
    - fovy: float, [batch], fov angle
    - aspect: float, [batch], aspect ratio. The ratio of width to height of the viewing window or viewport.
    - z_near, z_far: float, [batch], the z-clipping distances. These define the depth range of the scene that is visible. objects outside this range are clipped out. 

    Returns:
    - proj_mat: float, [batch x 4 x 4]
    """
    tan_half_fovy = torch.tan(fovy / 2.0)
    zeros_pl = torch.zeros_like(fovy)
    ones_pl = torch.ones_like(fovy)

    k1 = -(z_far + z_near) / (z_far - z_near)
    k2 = -2.0 * z_far * z_near / (z_far - z_near)
    return torch.stack(
        [
            1.0 / aspect / tan_half_fovy,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            1.0 / tan_half_fovy,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            k1,
            k2,
            zeros_pl,
            zeros_pl,
            -ones_pl,
            zeros_pl,
        ],
        -1,
    ).view(-1, 4, 4)


def orthorgonal(scalex, scaley, z_near, z_far):
    """orthorgonal
    Parallel lines in the world space remain parallel after projection. There's no convergence to a vanishing point.
    Inputs:
    - scalex, scaley: These determine how the x and y dimensions are scaled in the resulting image. T
    - z_near, z_far: float, [batch], the z-clipping distances. These define the depth range of the scene that is visible. objects outside this range are clipped out. 

    Returns:
    - proj_mat: float, [batch x 4 x 4]
    """
    zeros_pl = torch.zeros_like(z_near)
    ones_pl = torch.ones_like(z_near)

    k1 = -2.0 / (z_far - z_near)
    k2 = (z_far + z_near) / (z_far - z_near)
    return torch.stack(
        [
            scalex,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            scaley,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            k1,
            k2,
            zeros_pl,
            zeros_pl,
            zeros_pl,
            ones_pl,
        ],
        -1,
    ).view(-1, 4, 4)


def transform(matrix, points):
    """
    Inputs:
    - matrix: float, [npoints x 4 x 4]
    - points: float, [npoints x 3]

    Outputs:
    - transformed_points: float, [npoints x 3]
    """
    out = torch.cat([points, torch.ones_like(points[:, [0]], device=points.device)], dim=1).view(points.size(0), 4, 1)
    out = matrix @ out
    out = out[:, :3, 0] / out[:, [3], 0]    # the extra square bracket means that the dimension is kept
    return out


class ComputeDepthMaps(torch.nn.Module):
    def __init__(self, projection: str = "orthorgonal", eyepos_scale: float = 1.0, image_size: int = 256):
        super().__init__()

        self.image_size = image_size
        self.eyes_pos_list = [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
        self.num_views = len(self.eyes_pos_list)
        assert projection in {"perspective", "orthorgonal"}
        if projection == "perspective":
            self.projection_matrix = perspective(
                fovy=torch.tensor([torch.pi / 4], dtype=torch.float32),
                aspect=torch.tensor([1.0], dtype=torch.float32),
                z_near=torch.tensor([0.1], dtype=torch.float32),
                z_far=torch.tensor([10.0], dtype=torch.float32),
            )
        elif projection == "orthorgonal":
            self.projection_matrix = orthorgonal(
                scalex=torch.tensor([1.5], dtype=torch.float32),
                scaley=torch.tensor([1.5], dtype=torch.float32),
                z_near=torch.tensor([0.1], dtype=torch.float32),
                z_far=torch.tensor([10.0], dtype=torch.float32),
            )
        else:
            raise Exception("Unknown projection type")

        self.pre_matrix_list = []   # a list of total transformation matrices from multiple view points
        for i in range(self.num_views):
            _view_matrix = look_at(
                eyes=torch.tensor([self.eyes_pos_list[i]], dtype=torch.float32) * eyepos_scale,
                centers=torch.tensor([[0, 0, 0]], dtype=torch.float32),
                ups=torch.tensor([[0, 0, 1]], dtype=torch.float32),
            )

            self.register_buffer("_pre_matrix", self.projection_matrix @ _view_matrix)  # The final transformation matrix
            self.pre_matrix_list.append(self._pre_matrix)
            

    def forward(self, data, view_id=0, radius_list=[5]):
        if view_id >= self.num_views:
            return None

        _batch_size = data.size(0)
        _num_points = data.size(1)
        _matrix = self.pre_matrix_list[view_id].expand(_batch_size * _num_points, 4, 4).to(data.device) # originally was shape [1, 4, 4], expand to [B * N, 4, 4]
        _background = torch.zeros(_batch_size, 1, self.image_size, self.image_size, dtype=data.dtype, device=data.device)
        _batch_inds = torch.arange(0, _batch_size, dtype=torch.int32, device=data.device) # [B]
        _batch_inds = _batch_inds.unsqueeze(1).expand(_batch_size, _num_points).reshape(-1)  # [B, 1] -> [B, N] -> [B*N]

        pcds = data.view(-1, 3)  # [bs* num_points, 3]
        trans_pos = transform(_matrix, pcds)    # [B*N, 3]
        pos_xs, pos_ys, pos_zs = trans_pos.split(dim=1, split_size=1)
        # x, y as point pixel location
        pos_ijs = torch.cat([-pos_ys, pos_xs], dim=1)  # negate pos_ys because images row indices are from top to bottom
        # import matplotlib.pyplot as plt
        # plt.scatter(pos_ijs[:, 0].cpu().numpy(), pos_ijs[:, 1].cpu().numpy())
        # plt.show()
        # raise Exception("break")
        # z as point feature
        point_features = 1.0 - (pos_zs - pos_zs.min()) / (pos_zs.max() - pos_zs.min())  # npoints x 1, a one-channel point feature
        # depth_maps: [bs, 1, 256, 256]
        # depth_maps = (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min())
        for radius in radius_list:
            if radius == radius_list[0]:
                depth_maps = p2i(
                    pos_ijs,
                    point_features,
                    _batch_inds,
                    _background,
                    kernel_radius=radius_list[0],
                    kernel_kind_str="cos",
                    reduce="max",
                )
            else:
                _depth_maps = p2i(
                    pos_ijs,
                    point_features,
                    _batch_inds,
                    _background,
                    kernel_radius=radius,
                    kernel_kind_str="cos",
                    reduce="max",
                )
                depth_maps = torch.cat((depth_maps, _depth_maps), dim=1)
        return depth_maps


class PointCloud(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.compute_depth_maps = ComputeDepthMaps(projection="perspective", eyepos_scale=2, image_size=256).float().cuda()
        self.radius = 5
        # read point cloud
        ppcd = open3d.io.read_point_cloud("00.pcd")
        ppcd = np.asarray(ppcd.points)
        ppcd = torch.from_numpy(ppcd).float().cuda()   # make sure point cloud must be sent to gpu in order to use p2i_op kernel

        # normalize the point cloud
        ppcd = ppcd - ppcd.mean(dim=0, keepdim=True)
        ppcd = ppcd / ppcd.norm(dim=1, keepdim=True).max()
        self.fixed_pcd = ppcd.clone().detach()
        self.pcd = torch.nn.Parameter(ppcd.requires_grad_(True))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, eps=1e-15)
    def forward(self, view_matrix=torch.zeros((4, 4), device="cuda"), projection_matrix=torch.zeros((4, 4), device="cuda")):
        view_matrix = view_matrix.float().cuda()
        projection_matrix = projection_matrix.float().cuda()
        transform_matrix = projection_matrix @ view_matrix
        background = torch.zeros(1, 1, 256, 256, dtype=torch.float32, device=self.pcd.device)
        points_2d = transform(transform_matrix, torch.cat([self.pcd, self.fixed_pcd], dim=0))
        pos_xs, pos_ys, pos_zs = points_2d.split(dim=1, split_size=1)
        pos_ijs = torch.cat([pos_ys, pos_xs], dim=1)
        point_features = 1.0 - (pos_zs - pos_zs.min()) / (pos_zs.max() - pos_zs.min())
        img_2d = p2i(
            pos_ijs,
            point_features,
            torch.zeros(pos_ijs.shape[0], dtype=torch.int32, device=self.pcd.device),
            background,
            kernel_radius=self.radius,
            kernel_kind_str="cos",
            reduce="max",
        )
        
        return img_2d
    def save_ply(self):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(self.pcd.cpu().detach().numpy())
        open3d.io.write_point_cloud("00.ply", pcd)


        

        
        
    

if __name__ == "__main__":
    import os
    import numpy as np
    import open3d
    import torch
    import torchvision

    if not os.path.exists("__temp__"):
        os.mkdir("__temp__")
    model = PointCloud()

    eye_pos = torch.tensor([[-1, -1, -1]], dtype=torch.float32, device="cuda")
    view_matrix = look_at(
                eyes=torch.tensor([[-1, -1, -1]], dtype=torch.float32),
                centers=torch.tensor([[0, 0, 0]], dtype=torch.float32),
                ups=torch.tensor([[0, 0, 1]], dtype=torch.float32),
            )
    projection_matrix = perspective(
                fovy=torch.tensor([torch.pi / 4], dtype=torch.float32),
                aspect=torch.tensor([1.0], dtype=torch.float32),
                z_near=torch.tensor([0.1], dtype=torch.float32),
                z_far=torch.tensor([10.0], dtype=torch.float32),
            )
    

    depth_map = model(view_matrix, projection_matrix).squeeze()
    torchvision.utils.save_image(depth_map, "__temp__/tky.png", pad_value=1)
    # for i in range(8):
    #     torch.cuda.empty_cache()
    #     depth_map = model()
    #     print(torch.sum(depth_map))
    #     print(depth_map)
    #     print(depth_map.shape)
    #     torchvision.utils.save_image(depth_map, f"__temp__/depth_map_{i}.jpg", pad_value=1)

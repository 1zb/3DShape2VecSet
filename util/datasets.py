import torch

from .shapenet import ShapeNet

class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        
    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, point


def build_shape_surface_occupancy_dataset(split, args):
    if split == 'train':
        # transform = #transforms.Compose([
        transform = AxisScaling((0.75, 1.25), True)
        # ])
        return ShapeNet(args.data_path, split=split, transform=transform, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
    elif split == 'val':
        # return ShapeNet(args.data_path, split=split, transform=None, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
        return ShapeNet(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
    else:
        return ShapeNet(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)

if __name__ == '__main__':
    # m = ShapeNet('/home/zhanb0b/data/', 'train', transform=AxisScaling(), sampling=True, num_samples=1024, return_surface=True, surface_sampling=True)
    m = ShapeNet('/home/zhanb0b/data/', 'train', transform=AxisScaling(), sampling=True, num_samples=1024, return_surface=True, surface_sampling=True)
    p, l, s, c = m[0]
    print(p.shape, l.shape, s.shape, c)
    print(p.max(dim=0)[0], p.min(dim=0)[0])
    print(p[l==1].max(axis=0)[0], p[l==1].min(axis=0)[0])
    print(s.max(axis=0)[0], s.min(axis=0)[0])
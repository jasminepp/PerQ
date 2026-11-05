"""
Datasets
"""
from perq.utils import *
from perq.io_utils import *


class TrainingDatasetBase(torch.utils.data.Dataset):
    def __init__(self, logger, path, split, video_size=(-1, -1, -1), patch_size=(-1, -1, -1),
                 channel_scale=None, channel_shift=None):
        self.video_size = tuple(video_size[d] if video_size[d] != -1 else self.video_size[d] for d in range(3))
        self.patch_size = tuple(patch_size[d] if patch_size[d] != -1 else self.video_size[d] for d in range(3))
        assert len(self.video_size) == len(self.patch_size) == 3, \
            'video_size and patch_size should be 3D tuples'
        assert all(self.video_size[d] > 0 or self.video_size[d] == -1 for d in range(3)), \
            'video_size should be positive or -1'
        assert all(self.patch_size[d] > 0 or self.patch_size[d] == -1 for d in range(3)), \
            'patch_size should be positive or -1'

        self.logger = logger
        self.path = os.path.expanduser(path)
        self.split = split
        self.video_names = self._get_video_names()
        self.transform = self._get_transform()

        assert channel_scale is None or channel_shift is None or \
               (len(channel_scale) == self.get_num_channels() and len(channel_shift) == self.get_num_channels())
        self.channel_scale, self.channel_shift = compute_scale_shift(self.get_num_channels(), self.get_bit_depth(),
                                                                     channel_scale, channel_shift)

        self.logger.info(f'{type(self).__name__}:')
        self.logger.info(f'     Path: {self.path}')
        self.logger.info(f'     Split: {self.split}')
        self.logger.info(f'     Number of videos: {len(self)}')
        self.logger.info(f'     Video size: {self.get_video_size()}    Patch size: {self.get_patch_size()}')
        self.logger.info(f'     Channels: {self.get_num_channels()}    Bit depth: {self.get_bit_depth()}')
        self.logger.info(f'     Scale: {self.channel_scale}    Shift: {self.channel_shift}')

    def get_num_videos(self):
        return len(self)

    def get_video_size(self):
        return self.video_size

    def get_patch_size(self):
        return self.patch_size

    def get_num_channels(self):
        raise NotImplementedError

    def get_bit_depth(self):
        raise NotImplementedError

    def _get_video_names(self):
        raise NotImplementedError

    def _get_video_frame_paths(self, video_name):
        raise NotImplementedError

    def _get_transform(self):
        if self.get_patch_size()[1] == -1 and self.get_patch_size()[2] == -1:
            return torchvision.transforms.Compose([])
        elif self.get_patch_size()[1] != -1 and self.get_patch_size()[2] != -1:
            assert self.split == 'train', f'split must be "train" when patch_size is set, got {self.split!r}'
            return torchvision.transforms.Compose(
                [torchvision.transforms.RandomCrop(self.get_patch_size()[1:], pad_if_needed=True)]
            )
        else:
            raise ValueError('patch_size[1] and patch_size[2] should be both -1 or both positive')

    def _get_patch(self, vidx):
        # Sample random frames
        frame_paths = self._get_video_frame_paths(self.video_names[vidx])
        n_frames = self.get_patch_size()[0] if self.get_patch_size()[0] != -1 else len(frame_paths)
        if self.split == 'train':
            # Drawn for one-frame images too: the Python RNG stream depends on it
            start_frame_idx = random.randint(0, len(frame_paths) - n_frames)
        else:
            start_frame_idx = 0
        end_frame_idx = start_frame_idx + n_frames
        frames = torch.cat([torchvision.io.decode_image(torchvision.io.read_file(frame_path_i),
                                                        mode=torchvision.io.ImageReadMode.RGB)[:, None] \
                           for frame_path_i in frame_paths[start_frame_idx:end_frame_idx]], dim=1)

        # Sample random patches (training) or keep the full images (evaluation)
        patches = self.transform(frames).float()

        return self.channel_scale.view(self.get_num_channels(), 1, 1, 1) * patches + \
               self.channel_shift.view(self.get_num_channels(), 1, 1, 1)

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        vidx_max = torch.tensor(len(self), dtype=torch.int64)
        vidx = torch.tensor(idx, dtype=torch.int64)
        idx_thw_max = torch.tensor((1, 1, 1), dtype=torch.int64)
        idx_thw = torch.tensor((0, 0, 0), dtype=torch.int64)
        patch = self._get_patch(idx)
        return (vidx, vidx_max, idx_thw, idx_thw_max,
                torch.tensor(patch.shape[1:], dtype=torch.int64), torch.tensor(patch.shape[1:], dtype=torch.int64)), \
               torch.clip_(patch, 0., 1.)


class MultiImageDataset(TrainingDatasetBase):
    video_size = (-1, -1, -1)

    def get_num_channels(self):
        return 3

    def get_bit_depth(self):
        return 8

    def _get_video_names(self):
        assert os.path.isdir(self.path), f'{self.path} should be a directory'
        assert all(video_name.lower().endswith(('.png', '.jpg', '.jpeg')) for video_name in os.listdir(self.path)), \
            f'All files in {self.path} should be images'
        return sorted(os.listdir(self.path))

    def _get_video_frame_paths(self, video_name):
        return [os.path.join(self.path, video_name)]


def create_datasets(logger, dataset_cfgs,
                    channel_scale=None, channel_shift=None, split='train',
                    force_same_shape=False):
    """
    Image datasets: a sample is one image, randomly cropped to patch_size for training and kept whole for evaluation.
    Images of different sizes need batch size 1 when no patch_size is set.
    """
    datasets = []

    for cfg in dataset_cfgs:
        # Create dataset
        if cfg.dataset_type == 'multi_images':
            assert cfg.video_name is None, \
                f'video_name should be None for multi_images dataset'
            assert tuple(cfg.video_size) == (-1, -1, -1), \
                f'video_size should be (-1, -1, -1) for multi_images dataset'
            dataset = MultiImageDataset(logger, cfg.dataset_dir, video_size=cfg.video_size,
                                        patch_size=cfg.patch_size,
                                        channel_scale=channel_scale, channel_shift=channel_shift, split=split)
        else:
            raise ValueError(f'Invalid dataset type: {cfg.dataset_type}')
        datasets.append(dataset)

    # Check dataset shapes
    assert not force_same_shape or \
           all(datasets[0].get_video_size() == dataset.get_video_size() for dataset in datasets), \
        f'All datasets should have the same video_size when force_same_shape is True'
    assert not force_same_shape or \
           all(datasets[0].get_patch_size() == dataset.get_patch_size() for dataset in datasets), \
        f'All datasets should have the same patch_size when force_same_shape is True'

    return datasets
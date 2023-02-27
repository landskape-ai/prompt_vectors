import torch
import numpy as np


class PadPrompter(torch.torch.nn.Module):
    def __init__(self, args, identity=False):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        self.base_size = image_size - pad_size * 2
        tensor_class = torch.zeros if identity else torch.randn

        self.pad_up = torch.torch.nn.Parameter(tensor_class([1, 3, pad_size, image_size]))
        self.pad_down = torch.torch.nn.Parameter(tensor_class([1, 3, pad_size, image_size]))
        self.pad_left = torch.torch.nn.Parameter(
            tensor_class([1, 3, image_size - pad_size * 2, pad_size])
        )
        self.pad_right = torch.torch.nn.Parameter(
            tensor_class([1, 3, image_size - pad_size * 2, pad_size])
        )

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt


class FixedPatchPrompter(torch.nn.Module):
    def __init__(self, args, identity=False):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size

        tensor_class = torch.zeros if identity else torch.randn
        self.patch = torch.nn.Parameter(tensor_class([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, : self.psize, : self.psize] = self.patch

        return x + prompt


class RandomPatchPrompter(torch.nn.Module):
    def __init__(self, args, identity):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size

        tensor_class = torch.zeros if identity else torch.randn
        self.patch = torch.nn.Parameter(tensor_class([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, x_ : x_ + self.psize, y_ : y_ + self.psize] = self.patch

        return x + prompt


def padding(args, identity=False):
    return PadPrompter(args, identity)


def fixed_patch(args, identity=False):
    return FixedPatchPrompter(args, identity)


def random_patch(args, identity=False):
    return RandomPatchPrompter(args, identity)

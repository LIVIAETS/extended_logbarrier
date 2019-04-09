#!/usr/bin/env python3.6

from itertools import repeat
from typing import Any, Callable, List

import torch
from torch import Tensor

from utils import eq


class ConstantBounds():
    def __init__(self, **kwargs):
        self.C: int = kwargs['C']
        self.const: Tensor = torch.zeros((self.C, 1, 2), dtype=torch.float32)

        for i, (low, high) in kwargs['values'].items():
            self.const[i, 0, 0] = low
            self.const[i, 0, 1] = high

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        return self.const


class TagBounds(ConstantBounds):
    def __init__(self, **kwargs):
        super().__init__(C=kwargs['C'], values=kwargs["values"])  # We use it as a dummy

        self.idc: List[int] = kwargs['idc']
        self.ignore_disp: bool
        if 'ignore_disp' in kwargs:
            self.ignore_disp = kwargs['ignore_disp']
        else:
            self.ignore_disp = False
        self.idc_mask: Tensor = torch.zeros(self.C, dtype=torch.uint8)  # Useful to mask the class booleans
        self.idc_mask[self.idc] = 1

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        positive_class: Tensor = torch.einsum("cwh->c", target) > 0
        weak_positive_class: Tensor = torch.einsum("cwh->c", weak_target) > 0

        masked_positive: Tensor = torch.einsum("c,c->c", positive_class, self.idc_mask).type(torch.float32)  # Keep only the idc
        masked_weak: Tensor = torch.einsum("c,c->c", weak_positive_class, self.idc_mask).type(torch.float32)
        assert eq(masked_positive, masked_weak) or self.ignore_disp, f"Unconsistent tags between labels: {filename}"

        res: Tensor = super().__call__(image, target, weak_target, filename)
        masked_res = torch.einsum("cki,c->cki", res, masked_positive)

        return masked_res


class PreciseBounds():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']

        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        value: Tensor = self.__fn__(target[None, ...])[0].type(torch.float32)  # cwh and not bcwh
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        elif self.mode == "abs":
            margin = torch.ones_like(value) * self.margin
        else:
            raise ValueError("mode")

        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)

        return res


class PreciseTags(PreciseBounds):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.neg_value: Tensor = Tensor(kwargs['neg_value'])

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        positive_class: Tensor = torch.einsum("cwh->c", target) > 0

        res = super().__call__(image, target, weak_target, filename)
        _, k, two = res.shape
        assert self.neg_value.shape == (k, two)
        # if (positive_class == 0).sum():
        #     print("new", res.shape)
        #     print(res[1])

        masked = res[...]
        masked[positive_class == 0] = self.neg_value[...]
        assert masked.shape == res.shape
        # if (positive_class == 0).sum():
        #     print(masked[1, ..., 0])
        #     print(masked[1, ..., 1])
        # #     print(res[1])
        #     print(masked[1])
        #     exit(-1)

        return masked


class PreciseUpper(PreciseBounds):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        res = super().__call__(image, target, weak_target, filename)
        c, d, b = res.shape
        assert b == 2

        positive_class: Tensor = torch.einsum("cwh->c", target) > 0
        assert positive_class.shape == (c,)

        masked = res[...]
        masked[positive_class, :, 0] = 1
        masked[~positive_class, :, 0] = 0  # Probably superfluous

        return masked


class BoxBounds():
    def __init__(self, **kwargs):
        self.margins: Tensor = torch.Tensor(kwargs['margins'])
        assert len(self.margins) == 2
        assert self.margins[0] <= self.margins[1]

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        c = len(weak_target)
        box_sizes: Tensor = torch.einsum("cwh->c", weak_target)[..., None].type(torch.float32)

        bounds: Tensor = box_sizes * self.margins

        res = bounds[:, None, :]
        assert res.shape == (c, 1, 2)
        assert (res[..., 0] <= res[..., 1]).all()

        # exact_sizes: Tensor = torch.einsum("cwh->c", target).type(torch.float32)
        # assert (res[3, 0, 0] <= exact_sizes[3]).all(), (res[:, 0, 0], exact_sizes, box_sizes[..., 0])
        # assert (res[3, 0, 1] >= exact_sizes[3]).all(), (res[:, 0, 1], exact_sizes, box_sizes[..., 0])

        return res


class PredictionBounds():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']

        # Do it on CPU to avoid annoying the main loop
        self.net: Callable[Tensor, [Tensor]] = torch.load(kwargs['net'], map_location='cpu')

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        with torch.no_grad():
            value: Tensor = self.net(image[None, ...])[0].type(torch.float32)[..., None]  # cwh and not bcwh
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        elif self.mode == "abs":
            margin = torch.ones_like(value) * self.margin
        else:
            raise ValueError("mode")

        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)

        return res


class TagsPredictions(PredictionBounds):
    """
    Put the boudns value to neg_value for the negative classes
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.neg_value: Tensor = Tensor(kwargs['neg_value'])

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        positive_class: Tensor = torch.einsum("cwh->c", target) > 0

        res = super().__call__(image, target, weak_target, filename)
        _, k, two = res.shape
        assert self.neg_value.shape == (k, two)
        # if (positive_class == 0).sum():
        #     print("new", res.shape)
        #     print(res[1])

        masked = res[...]
        masked[positive_class == 0] = self.neg_value[...]
        assert masked.shape == res.shape
        # if (positive_class == 0).sum():
        #     print(masked[1, ..., 0])
        #     print(masked[1, ..., 1])
        # #     print(res[1])
        #     print(masked[1])
        #     exit(-1)

        return masked

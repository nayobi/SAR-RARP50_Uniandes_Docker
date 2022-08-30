# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Any, List, Tuple, Optional

import torch
from torch import device
from torch import nn
from torch.nn import functional as F
from modeling import D2SwinTransformer, MaskFormerHead

def shapes_to_tensor(x: List[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Turn a list of integer scalars or integer Tensor scalars into a vector,
    in a way that's both traceable and scriptable.

    In tracing, `x` should be a list of scalar Tensor, so the output can trace to the inputs.
    In scripting or eager, `x` should be a list of int.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        assert all(
            [isinstance(t, torch.Tensor) for t in x]
        ), "Shape should be tensor during tracing!"
        # as_tensor should not be used in tracing because it records a constant
        ret = torch.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret
    return torch.as_tensor(x, device=device)

class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size.
    The original sizes of each image is stored in `image_sizes`.

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w).
            During tracing, it becomes list[Tensor] instead.
    """

    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
                be smaller than (H, W) due to padding.
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Access the individual image in its original size.

        Args:
            idx: int or slice

        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self) -> device:
        return self.tensor.device

    @staticmethod
    def from_tensors(
        tensors: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0
    ) -> "ImageList":
        """
        Args:
            tensors: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
                to the same shape with `pad_value`.
            size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
                the common height and width is divisible by `size_divisibility`.
                This depends on the model and many models need a divisibility of 32.
            pad_value (float): value to pad

        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        image_sizes_tensor = [shapes_to_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

        # handle weirdness of scripting and tracing ...
        if torch.jit.is_scripting():
            max_size: List[int] = max_size.to(dtype=torch.long).tolist()
        else:
            if torch.jit.is_tracing():
                image_sizes = image_sizes_tensor

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
        else:
            # max_size can be a tensor in tracing mode, therefore convert to list
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)

def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result

class MaskFormer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        backbone = D2SwinTransformer(cfg)
        sem_seg_head = MaskFormerHead(cfg, backbone.output_shape())
        num_queries= cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        object_mask_threshold= cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD
        overlap_threshold= cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD
        size_divisibility= cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        pixel_mean= cfg.MODEL.PIXEL_MEAN
        pixel_std= cfg.MODEL.PIXEL_STD
        instance_on= cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
        test_topk_per_image= cfg.TEST.DETECTIONS_PER_IMAGE

        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = True
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.instance_on = instance_on
        self.test_topk_per_image = test_topk_per_image
        assert self.sem_seg_postprocess_before_inference

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        # breakpoint()
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        # breakpoint()

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        mask_embd_results = outputs["mask_embeds"]
        # decod_out_results = outputs["decoder_outputs"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, mask_embd, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, mask_embd_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})
            # additions.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)
            
            # instance segmentation inference
            if self.instance_on:
                instance_r = self.instance_inference(mask_cls_result, mask_pred_result, mask_embd)
                processed_results[-1]["instances"] = instance_r

        return processed_results

    def instance_inference(self, mask_cls, mask_pred, mask_embd):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        # breakpoint()

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // (self.sem_seg_head.num_classes)
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]
        mask_embd = mask_embd[topk_indices]
        # decod_out = decod_out[topk_indices]
        score_dist = scores[topk_indices]

        result = {} #Instances(image_size)
        # mask (before sigmoid)
        bin_mask = (mask_pred > 0).float()
        result['pred_masks'] = bin_mask #(mask_pred > 0).float()
        result['mask_embed'] = mask_embd
        # result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result['pred_boxes'] = BitMasks(mask_pred > 0).get_bounding_boxes()
        result['score_dist'] = score_dist

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result['pred_masks'].flatten(1)).sum(1) / (result['pred_masks'].flatten(1).sum(1) + 1e-6)
        result['scores'] = scores_per_image * mask_scores_per_image
        result['pred_classes'] = labels_per_image
        return result

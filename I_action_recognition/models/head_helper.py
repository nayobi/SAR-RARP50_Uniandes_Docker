#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn

class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)
        return x


class TransformerRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        cfg,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,

    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(TransformerRoIHead, self).__init__()
        self.cfg = cfg
        
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.multi_faster = nn.Parameter(torch.tensor([0.0]),requires_grad=False)
        
        
        dim_faster = 1024
        dim_final = 2048

        if cfg.FASTER.TRANS:
            self.trans_project = nn.Linear(768,dim_faster,bias=False)
            self.mlp = nn.Sequential(nn.Linear(256, dim_faster, bias=False),
                                    # nn.ReLU(),
                                    # nn.Linear(dim_faster, dim_faster, bias=False),
                                    nn.BatchNorm1d(dim_faster))
        else:
            self.mlp = nn.Sequential(nn.Linear(256, dim_final - 768, bias=False),
                                    # nn.ReLU(),
                                    # nn.Linear(dim_final - 768, dim_final - 768, bias=False),
                                    nn.BatchNorm1d(dim_final -768))
        
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Sequential(nn.Linear(dim_final, num_classes, bias=True),)
                                        # nn.ReLU(),
                                        # nn.Linear(256, num_classes, bias=True))
        self.act_func = act_func
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, thw, bboxes, features=None):
 
        x = inputs.mean(1)
        if self.cfg.FASTER.TRANS:
            x = self.trans_project(x)
        x_boxes = torch.zeros(len(bboxes), x.shape[1], device = inputs.device, requires_grad=True)
        for i in range(len(inputs)):
            x_boxes[bboxes[:,0] == i].copy_(x[i])
            # x[i].detach()

        features = features[:,1:]
        features = self.mlp(features)
        x = torch.cat([x_boxes, features], dim=1)
        
        ##########################

        x = self.projection(x)
        if not self.training:
            x = self.act(x)

        return x, self.multi_faster
        

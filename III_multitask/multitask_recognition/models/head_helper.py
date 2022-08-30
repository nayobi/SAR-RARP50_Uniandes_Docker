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

        super(TransformerRoIHead, self).__init__()
        self.cfg = cfg
        
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # self.multi_faster = nn.Parameter(torch.tensor([0.0]),requires_grad=False)
        
        
        dim_faster = 1024
        dim_final = 2048

        input_dim = 256
        

        if cfg.FASTER.TRANS:
            self.trans_project = nn.Linear(768,dim_faster,bias=True)
            self.mlp = nn.Sequential(nn.Linear(input_dim, dim_faster, bias=True),
                                    nn.ReLU(),)
        else:
            self.mlp = nn.Sequential(nn.Linear(input_dim, dim_final - 768, bias=True),
                                    nn.ReLU(),)
        
        self.projection = nn.Sequential(nn.Linear(dim_final, num_classes, bias=True),)

        self.act = nn.Softmax(dim=1)


    def forward(self, inputs,  features, idx):
        x = inputs
        x = inputs.mean(1)
        if self.cfg.FASTER.TRANS:
            x = self.trans_project(x)
        x_boxes = torch.zeros(len(features), x.shape[1], device = inputs.device, requires_grad=True)
        for i in range(len(inputs)):
            x_boxes[idx==i].copy_(x[i])

        features = self.mlp(features)
        x = torch.cat([x_boxes, features], dim=1)

        x = self.projection(x)
        x = self.act(x)

        return x
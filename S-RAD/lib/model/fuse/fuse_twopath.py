import torch
import torch.nn as nn

class Fuse_twopath(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(Fuse_twopath, self).__init__()
        self.conv_1x1 = nn.Conv2d(
            dim_in*3,
            dim_in ,
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(
            num_features=dim_in)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x_1,x_2):
        feature_h = x_1.shape[2]
        feature_w = x_1.shape[3]
        batch = x_1.shape[0]

        x_2 = x_2.view(batch,-1,feature_h,feature_w)
        x_fuse = torch.cat([x_1, x_2], 1)
        x_fuse = self.conv_1x1(x_fuse)
        x_fuse = self.bn(x_fuse)
        x_fuse = self.relu(x_fuse)
        return x_fuse
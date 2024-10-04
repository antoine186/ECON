# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.net.BasePIFuNet import BasePIFuNet
from lib.net.FBNet import GANLoss, IDMRFLoss, VGGLoss, define_D, define_G
from lib.net.net_util import init_net

# This is Gn_front and Gn_back.
# Inputs are likely: RGB Image + SMPL Prior.
class NormalNet(BasePIFuNet):
    """
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    """
    def __init__(self, cfg):

        super(NormalNet, self).__init__()

        # This cfg.net configuration seems to carry parameters not just meant for NormalNet.

        # HGPIFuNet: (Hourglass PIFuNet) is used to predict the detailed surface normal maps for the clothed human.
            # Inputs to HGPIFuNet: RGB Image + SMPL Prior.
        # SDF: (Signed distance function) set to False.
        # ResNet-18: ResNet-18 is used to extract features (multi-person segmentation) from the input image before predicting the normal maps.
            # Differs from paper apparently where they use ResNet-50 for multi-person segmentation either before 
            # normal map predictions or SMPL prior prediction.
        # classifierIMF: 'MultiSegClassifier'.
            # IMF stands for intermediate map feature?
            # In the ECON pipeline, the network needs to be able to identify various regions of the human body and clothing.

            # The classifier is designed to handle multiple segmentation tasks simultaneously. This would involve segmenting 
            # different regions or aspects of the image, such as separating the body parts (e.g., head, arms, torso) or 
            # different garment types (e.g., shirt, pants).
        # Group normalisation: a normalisation technique applied to the intermediate layers of the network. This helps stabilize the training process.
        # Group normalisation: applied in the MLP (Multi-Layer Perceptron) part of the network.
            # The use of an MLP (Multi-Layer Perceptron) in predicting clothed normal maps is a common design choice in neural networks, 
            # particularly for dense pixel-wise prediction tasks like normal map estimation.
        # hg_down: 'ave_pool'.
            # This indicates that average pooling is used as a down-sampling method in the Hourglass network. This pooling strategy may help retain 
            # global information while reducing the spatial resolution, which is critical in hierarchical feature extraction.
        # conv1: [7, 2, 1, 3].
            # Parameters for the first convolutional layer: a 7x7 kernel, stride of 2, padding of 1, and 3 input channels (RGB image).
        # conv3x3: [3, 1, 1, 1]. <- Intermediate hourglass C layer.
            # Parameters for 3x3 convolutions: kernel size 3, stride 1, padding 1, and 1 input channel (likely for feature maps).
        # num_stack: 4.
            # Indicates the network uses 4 stacks of the Hourglass structure. Stacking multiple Hourglass networks is common for refining predictions 
            # at multiple scales.
        # num_hourglass: 2.
            # Specifies 2 Hourglass modules are used in the network for refining predictions, potentially referring to the front and back normal maps 
            # of the human body.
        # hourglass_dim: 256.
            # The dimension of the Hourglass feature map is set to 256, probably determining the depth of feature representation as this is different 
            # from the 128x128 or 512x512 resolution of PiFU image encodings.
        # voxel_dim: 32
            # Resolution of the voxel grid used in 3D reconstruction.
            # Instead of randomly sampling points in space, the voxel grid can provide a structured set of points where the implicit function is queried.
        # resnet_dim: 120
            # The dimension is set to 120, probably determining the depth of feature representation.
        # mlp_dim: [320, 1024, 512, 256, 128, 1]
            # The dimension of the MLP layers. This is clearly an implicit function given the 1-dimensional output.
        # mlp_dim_knn: [320, 1024, 512, 256, 128, 3]
            # Don't know what this does as knn not mentioned in the ECON paper.
        # mlp_dim_multiseg: [1088, 2048, 1024, 500]
            # MLP used for multi-segmentation tasks. Not sure why this is needed? Does it clash with ResNet-18 that seems to do the same segmentation task? Or
            # is it instead up or downstream from it?
        # res_layers: [2, 3, 4]
            # Specifies which layers of the ResNet backbone are used for extracting features.
        # filter_dim: 256
            # The dimension/number of feature filters used in the 3D processing network.
            # Could be a setting used to extract Fp, the mesh-based local feature vectors.
        # smpl_dim: 3
            # The SMPL dimensionality used in the network (3D for position or normal vectors).
        # cly_dim: 3
            # Clothing features are set to 3 for 3D.
        # soft_dim: 64
            # Soft body deformation dimension, which might be used to handle soft garments.
        # z_size: 200.0
            # The size of the latent space used in the latent feature representation.
        # N_freqs: 10
            # Refers to the number of frequencies used in positional encoding.
            # Positional encoding helps by transforming the input coordinates (e.g., 3D positions) into a higher-dimensional space, making it easier for the network 
            # to learn high-frequency details.
        # geo_w: 0.1
            # Weight for the geometric consistency loss, likely ensuring that the predicted normals are geometrically consistent with the SMPL body mesh.
        # norm_w: 0.1
            # Weight for the normal loss, which ensures that the predicted normal maps are consistent with ground-truth normals.
        # dc_w: 0.1
            # Weight for the data consistency loss, likely ensuring consistency between predicted and ground-truth 3D representations.
        # C_cat_to_G: False
            # A flag that may refer to whether to concatenate additional features (e.g., color) into the generator's input.
        # skip_hourglass: True
            # Indicates whether to allow skip connections in the Hourglass network, which can help preserve spatial information across different scales.
        # use_tanh: True
            # Indicates that tanh activation is used, likely for the normal maps to normalize the output between -1 and 1.
        # soft_onehot: True
            # Soft one-hot encoding, which might be used for segmentation purposes.
        # no_residual: True
            # Disables residual connections in certain parts of the network.
        # use_attention: False
            # Specifies that attention mechanisms are not used in this network.
        # prior_type: 'icon'
            # Specifies that the network uses an ICON prior. Hopefully, this means that ECON uses the ICON body refinement loop.
            # Otherwise, if it uses the ICON reconstruction as a prior, it probably clashes with the SMPL-X prior and thus makes
            # things more confusing.
        # smpl_feats: ['sdf', 'vis']
            # Defines the features used from the SMPL model — SDF (Signed Distance Function) and visibility.
            # This is confusing because while present in ICON, those functions are not explicitly mentioned in ECON.
        # use_filter: True
            # Indicates that some form of filtering is applied to the input data or the output predictions.
        # use_cc: False
            # Refers to whether color correction is used in the pipeline, which is set to False.
        # use_PE: False
            # Refers to whether positional encoding (PE) is used for the network's inputs. Set to False!
        # use_IGR: False
            # Indicates whether Implicit Geometric Regularization (IGR) is used. For instance, is SDF applied or not?
        # use_gan: False
            # Indicates that GAN (Generative Adversarial Networks) are not being used for this specific configuration.
            # What's a GAN doing here? Very weird.
        self.opt = cfg.net

        # There are no losses here.
        # All are zero length lists.
        self.F_losses = [item[0] for item in self.opt.front_losses]
        self.B_losses = [item[0] for item in self.opt.back_losses]
        self.F_losses_ratio = [item[1] for item in self.opt.front_losses]
        self.B_losses_ratio = [item[1] for item in self.opt.back_losses]
        self.ALL_losses = self.F_losses + self.B_losses

        # training = True.
        # We skip all loss settings here.
        if self.training:
            if 'vgg' in self.ALL_losses:
                self.vgg_loss = VGGLoss()
            if ('gan' in self.ALL_losses) or ('gan_feat' in self.ALL_losses):
                self.gan_loss = GANLoss(use_lsgan=True)
            if 'mrf' in self.ALL_losses:
                self.mrf_loss = IDMRFLoss()
            if 'l1' in self.ALL_losses:
                self.l1_loss = nn.SmoothL1Loss()

        # Specifies the input to Gn_front: "image" + "T_normal_F".
        self.in_nmlF = [
            item[0] for item in self.opt.in_nml if "_F" in item[0] or item[0] == "image"
        ]
        # Specifies the input to Gn_front: "image" + "T_normal_B".
        self.in_nmlB = [
            item[0] for item in self.opt.in_nml if "_B" in item[0] or item[0] == "image"
        ]
        # Input dimension is 6 because self.opt.in_nml = [('image', 3), ('T_normal_F', 3), ('T_normal_B', 3)]
        # and we only want the input image channels and the front body normal map channels.
        self.in_nmlF_dim = sum([
            item[1] for item in self.opt.in_nml if "_F" in item[0] or item[0] == "image"
        ])
        # Same thing for the back input dimension.
        self.in_nmlB_dim = sum([
            item[1] for item in self.opt.in_nml if "_B" in item[0] or item[0] == "image"
        ])

        # Creates and shapes the global generator for the front and the back.
        # Initialises netG's (i.e. the global generator) weights.

        # Input order:
            # Input number of channels: 6
            # Output number of channels: 3
            # Number of generator filters (or number of feature maps): 64
            # Do we use global or local generator: We use global generator
                # The GlobalGenerator focuses on producing a coarse global structure of the clothed normal map. It operates on the entire 
                # image at a relatively lower resolution and is responsible for capturing and generating the overall shape, structure, 
                # and global features of the image.
                    # Coarse Features: It focuses on large-scale features and ensures the consistency of the image at a macro level 
                    # (such as the overall shape of a person or object).
            # The number of downsampling layers used in the GlobalGenerator network: 4
            # The number of ResNet blocks to be used in the GlobalGenerator network: 9
            # n_local_enhancers: 1, irrelevant as not used
            # n_blocks_local: 3, irrelevant as not used
            # Image generation normalisation type: Instance normalisation, which is often used in image generation tasks as it normalises each instance separately.
        self.netF = define_G(self.in_nmlF_dim, 3, 64, "global", 4, 9, 1, 3, "instance")
        self.netB = define_G(self.in_nmlB_dim, 3, 64, "global", 4, 9, 1, 3, "instance")

        # SKIPPED
        if ('gan' in self.ALL_losses):
            self.netD = define_D(3, 64, 3, 'instance', False, 2, 'gan_feat' in self.ALL_losses)

        # Initialise a network: 1. register CPU/GPU device (with multi-GPU support); 2. Initialise the network weights.
        init_net(self)

    def forward(self, in_tensor):

        inF_list = []
        inB_list = []

        for name in self.in_nmlF:
            inF_list.append(in_tensor[name])
        for name in self.in_nmlB:
            inB_list.append(in_tensor[name])

        nmlF = self.netF(torch.cat(inF_list, dim=1))
        nmlB = self.netB(torch.cat(inB_list, dim=1))

        # ||normal|| == 1
        nmlF_normalized = nmlF / torch.norm(nmlF, dim=1, keepdim=True)
        nmlB_normalized = nmlB / torch.norm(nmlB, dim=1, keepdim=True)

        # output: float_arr [-1,1] with [B, C, H, W]
        mask = ((in_tensor["image"].abs().sum(dim=1, keepdim=True) != 0.0).detach().float())

        return nmlF_normalized * mask, nmlB_normalized * mask

    def get_norm_error(self, prd_F, prd_B, tgt):
        """calculate normal loss

        Args:
            pred (torch.tensor): [B, 6, 512, 512]
            tagt (torch.tensor): [B, 6, 512, 512]
        """

        tgt_F, tgt_B = tgt["normal_F"], tgt["normal_B"]

        # netF, netB, netD
        total_loss = {"netF": 0.0, "netB": 0.0}

        if 'l1' in self.F_losses:
            l1_F_loss = self.l1_loss(prd_F, tgt_F)
            total_loss["netF"] += self.F_losses_ratio[self.F_losses.index('l1')] * l1_F_loss
            total_loss["l1_F"] = self.F_losses_ratio[self.F_losses.index('l1')] * l1_F_loss
        if 'l1' in self.B_losses:
            l1_B_loss = self.l1_loss(prd_B, tgt_B)
            total_loss["netB"] += self.B_losses_ratio[self.B_losses.index('l1')] * l1_B_loss
            total_loss["l1_B"] = self.B_losses_ratio[self.B_losses.index('l1')] * l1_B_loss

        if 'vgg' in self.F_losses:
            vgg_F_loss = self.vgg_loss(prd_F, tgt_F)
            total_loss["netF"] += self.F_losses_ratio[self.F_losses.index('vgg')] * vgg_F_loss
            total_loss["vgg_F"] = self.F_losses_ratio[self.F_losses.index('vgg')] * vgg_F_loss
        if 'vgg' in self.B_losses:
            vgg_B_loss = self.vgg_loss(prd_B, tgt_B)
            total_loss["netB"] += self.B_losses_ratio[self.B_losses.index('vgg')] * vgg_B_loss
            total_loss["vgg_B"] = self.B_losses_ratio[self.B_losses.index('vgg')] * vgg_B_loss

        scale_factor = 0.5
        if 'mrf' in self.F_losses:
            mrf_F_loss = self.mrf_loss(
                F.interpolate(prd_F, scale_factor=scale_factor, mode='bicubic', align_corners=True),
                F.interpolate(tgt_F, scale_factor=scale_factor, mode='bicubic', align_corners=True)
            )
            total_loss["netF"] += self.F_losses_ratio[self.F_losses.index('mrf')] * mrf_F_loss
            total_loss["mrf_F"] = self.F_losses_ratio[self.F_losses.index('mrf')] * mrf_F_loss
        if 'mrf' in self.B_losses:
            mrf_B_loss = self.mrf_loss(
                F.interpolate(prd_B, scale_factor=scale_factor, mode='bicubic', align_corners=True),
                F.interpolate(tgt_B, scale_factor=scale_factor, mode='bicubic', align_corners=True)
            )
            total_loss["netB"] += self.B_losses_ratio[self.B_losses.index('mrf')] * mrf_B_loss
            total_loss["mrf_B"] = self.B_losses_ratio[self.B_losses.index('mrf')] * mrf_B_loss

        if 'gan' in self.ALL_losses:

            total_loss["netD"] = 0.0

            pred_fake = self.netD.forward(prd_B)
            pred_real = self.netD.forward(tgt_B)
            loss_D_fake = self.gan_loss(pred_fake, False)
            loss_D_real = self.gan_loss(pred_real, True)
            loss_G_fake = self.gan_loss(pred_fake, True)

            total_loss["netD"] += 0.5 * (loss_D_fake + loss_D_real
                                        ) * self.B_losses_ratio[self.B_losses.index('gan')]
            total_loss["D_fake"] = loss_D_fake * self.B_losses_ratio[self.B_losses.index('gan')]
            total_loss["D_real"] = loss_D_real * self.B_losses_ratio[self.B_losses.index('gan')]

            total_loss["netB"] += loss_G_fake * self.B_losses_ratio[self.B_losses.index('gan')]
            total_loss["G_fake"] = loss_G_fake * self.B_losses_ratio[self.B_losses.index('gan')]

            if 'gan_feat' in self.ALL_losses:
                loss_G_GAN_Feat = 0
                for i in range(2):
                    for j in range(len(pred_fake[i]) - 1):
                        loss_G_GAN_Feat += self.l1_loss(pred_fake[i][j], pred_real[i][j].detach())
                total_loss["netB"] += loss_G_GAN_Feat * self.B_losses_ratio[
                    self.B_losses.index('gan_feat')]
                total_loss["G_GAN_Feat"] = loss_G_GAN_Feat * self.B_losses_ratio[
                    self.B_losses.index('gan_feat')]

        return total_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import voxelmorph
from models.VoxelMorph_ import VoxelMorph, VoxelMorph2, VoxelMorph3, VoxelMorph22
from Source.loss_func import Grad
from Source.balance_func import ProcrustesSolver
from models.sxc_sp_cvpr2018 import cvpr2018_net1


def load_model(cfg):
    if cfg['ModelType'] == 'VoxelMorph':
        model = voxelmorph(
            backbone=cfg['BackBone'],
            feat_num=cfg['FeatNum'],
            img_size=cfg['ImgSize'],
            integrate_cfg=cfg['IntegrateConfig']
        )
    elif cfg['ModelType'] == 'net1':
        nf_enc = [16, 32, 32, 32]
        if cfg['Model'] == "vm1":
            nf_dec = [32, 32, 32, 32, 8, 8]
        elif cfg['Model'] == "vm2":
            nf_dec = [32, 32, 32, 32, 32, 16, 16]
        else:
            raise ValueError("Not yet implemented!")
        model = cvpr2018_net1(cfg['Vol_size'], nf_enc, nf_dec)
    else:
        raise NotImplementedError

    return model


def load_model1(cfg):
    if cfg['ModelType'] == 'VoxelMorph':
        model = VoxelMorph(
            backbone=cfg['BackBone'],
            feat_num=cfg['FeatNum'],
            img_size=cfg['ImgSize'],
            integrate_cfg=cfg['IntegrateConfig']
        )
    elif cfg['ModelType'] == 'net1':
        nf_enc = [16, 32, 32, 32]
        if cfg['Model'] == "vm1":
            nf_dec = [32, 32, 32, 32, 8, 8]
        elif cfg['Model'] == "vm2":
            nf_dec = [32, 32, 32, 32, 32, 16, 16]
        else:
            raise ValueError("Not yet implemented!")
        model = cvpr2018_net1(cfg['Vol_size'], nf_enc, nf_dec)
    else:
        raise NotImplementedError

    return model


def load_model2(cfg, model1):
    if cfg['ModelType'] == 'VoxelMorph':
        model = VoxelMorph2(
            backbone=cfg['BackBone'],
            feat_num=cfg['FeatNum'],
            img_size=cfg['ImgSize'],
            integrate_cfg=cfg['IntegrateConfig'],
            model1=model1
        )
    elif cfg['ModelType'] == 'net1':
        nf_enc = [16, 32, 32, 32]
        if cfg['Model'] == "vm1":
            nf_dec = [32, 32, 32, 32, 8, 8]
        elif cfg['Model'] == "vm2":
            nf_dec = [32, 32, 32, 32, 32, 16, 16]
        else:
            raise ValueError("Not yet implemented!")
        model = cvpr2018_net1(cfg['Vol_size'], nf_enc, nf_dec)
    else:
        raise NotImplementedError

    return model


def load_model3(cfg, model1, model2):
    if cfg['ModelType'] == 'VoxelMorph':
        model = VoxelMorph3(
            backbone=cfg['BackBone'],
            feat_num=cfg['FeatNum'],
            img_size=cfg['ImgSize'],
            integrate_cfg=cfg['IntegrateConfig'],
            model1=model1,
            model2=model2
        )
    elif cfg['ModelType'] == 'net1':
        nf_enc = [16, 32, 32, 32]
        if cfg['Model'] == "vm1":
            nf_dec = [32, 32, 32, 32, 8, 8]
        elif cfg['Model'] == "vm2":
            nf_dec = [32, 32, 32, 32, 32, 16, 16]
        else:
            raise ValueError("Not yet implemented!")
        model = cvpr2018_net1(cfg['Vol_size'], nf_enc, nf_dec)
    else:
        raise NotImplementedError

    return model


def load_model22(cfg, model1, model2, model3):
    if cfg['ModelType'] == 'VoxelMorph':
        model = VoxelMorph22(
            backbone=cfg['BackBone'],
            feat_num=cfg['FeatNum'],
            img_size=cfg['ImgSize'],
            integrate_cfg=cfg['IntegrateConfig'],
            model1=model1,
            model2=model2,
            model3=model3
        )
    elif cfg['ModelType'] == 'net1':
        nf_enc = [16, 32, 32, 32]
        if cfg['Model'] == "vm1":
            nf_dec = [32, 32, 32, 32, 8, 8]
        elif cfg['Model'] == "vm2":
            nf_dec = [32, 32, 32, 32, 32, 16, 16]
        else:
            raise ValueError("Not yet implemented!")
        model = cvpr2018_net1(cfg['Vol_size'], nf_enc, nf_dec)
    else:
        raise NotImplementedError

    return model


class GradientDescent(object):
    def __init__(self, model, optimizer, sim_loss, reg_loss, loss_weight, gradient_surgery):
        self.model = model
        self.optimizer = optimizer

        if sim_loss == 'MSE':
            self.similarity_cost = nn.MSELoss()
        elif sim_loss == 'MAE':
            self.similarity_cost = nn.L1Loss()
        else:
            raise NotImplementedError

        if reg_loss == 'Grad':
            self.smoothness_cost = Grad(penalty='l2')
        else:
            raise NotImplementedError

        self.loss_weight = loss_weight
        self.gradient_surgery = gradient_surgery
        self.scale_mode = 'rmse'

    def update_gradient(self, fwd):
        self.optimizer.zero_grad()
        similarity_loss = self.similarity_cost(fwd['Moved'], fwd['Fixed'])
        similarity_loss.backward(retain_graph=True)
        similarity_gradient = self.model.get_gradient()

        self.optimizer.zero_grad()
        smoothness_loss = self.smoothness_cost(fwd['Flow'])
        smoothness_loss.backward()
        smoothness_gradient = self.model.get_gradient()

        print_info = 'Loss ==> Similarity: {:.5f} Smoothness: {:.5f} '.format(
            similarity_loss.item(), smoothness_loss.item())

        loss_info = {
            'Similarity': similarity_loss.item(),
            'Smoothness': smoothness_loss.item(),
        }

        aggregated_similarity_gradient = torch.cat(similarity_gradient)
        aggregated_smoothness_gradient = torch.cat(smoothness_gradient)

        signed_similarity_gradient = torch.sign(aggregated_similarity_gradient)
        signed_smoothness_gradient = torch.sign(aggregated_smoothness_gradient)

        overlap = signed_similarity_gradient == signed_smoothness_gradient
        overlap_ratio = torch.sum(overlap) * 100 / aggregated_similarity_gradient.size(-1)
        print_info += 'Overlap: {:.2f}% '.format(overlap_ratio.item())
        loss_info['OverlapRatio'] = overlap_ratio.item()

        gradient = self._compute_gradient(fwd, aggregated_similarity_gradient, aggregated_smoothness_gradient,
                                          similarity_gradient, smoothness_gradient, overlap)

        self.model.set_gradient(gradient)
        self.optimizer.step()

        return similarity_loss.item(), loss_info, print_info

    def _compute_gradient(self, fwd, agg_sim, agg_smooth, sim_grad, smooth_grad, overlap):
        if self.gradient_surgery is None:
            if 'Lambda' in fwd.keys():
                gradient = 1 / (0.05 ** 2) * agg_sim + fwd['Lambda'].item() * agg_smooth
            else:
                gradient = self.loss_weight['Similarity'] * agg_sim + self.loss_weight['Smoothness'] * agg_smooth

        elif self.gradient_surgery == 'AgrSum':
            gradient = overlap * agg_sim
        elif self.gradient_surgery == 'AgrRand':
            gradient = overlap * agg_sim
            rand_gradient = torch.randn_like(gradient, device=gradient.device)
            rand_gradient *= ~overlap
            scale = gradient.abs().mean()
            rand_gradient *= scale
            gradient += rand_gradient
        elif self.gradient_surgery == 'PCGrad':
            gradient = self.__pcgrad(agg_sim, agg_smooth)
        elif self.gradient_surgery == 'OrsGrad':
            cosine = F.cosine_similarity(agg_sim, agg_smooth, dim=0)
            if cosine < 0:
                resize_gradient = self.Cross(agg_smooth, agg_sim)
                gradient = resize_gradient + agg_sim
            else:
                gradient = agg_sim
        elif self.gradient_surgery == 'OrrGrad':
            cosine = F.cosine_similarity(agg_sim, agg_smooth, dim=0)
            if cosine < 0:
                resize_gradient = self.Cross(agg_sim, agg_smooth)
                gradient = resize_gradient + agg_smooth
            else:
                gradient = agg_sim
        elif self.gradient_surgery == 'RtGrad':
            cosine = F.cosine_similarity(agg_sim, agg_smooth, dim=0)
            if cosine < 0:
                inner_prod = torch.dot(agg_sim, agg_smooth)
                proj_gradient = inner_prod / torch.linalg.vector_norm(agg_smooth).pow(2) * agg_smooth
                norm_gradient = agg_sim - proj_gradient
                scale_factor = torch.linalg.norm(agg_sim) / torch.linalg.norm(norm_gradient)
                gradient = scale_factor * norm_gradient
            else:
                gradient = agg_sim
        elif self.gradient_surgery in ['LayerWise', 'LayerWise_reg']:
            gradient = []
            for s_grad, r_grad in zip(sim_grad, smooth_grad):
                cosine = F.cosine_similarity(s_grad, r_grad, dim=0)
                if cosine < 0:
                    inner_prod = torch.dot(s_grad, r_grad)
                    if self.gradient_surgery == 'LayerWise':
                        proj_gradient = inner_prod / torch.linalg.vector_norm(r_grad).pow(2) * r_grad
                        norm_gradient = s_grad - proj_gradient
                    else:
                        proj_gradient = inner_prod / torch.linalg.vector_norm(s_grad).pow(2) * s_grad
                        norm_gradient = r_grad - proj_gradient
                    gradient.append(norm_gradient)
                else:
                    gradient.append(s_grad)
            gradient = torch.cat(gradient)
        elif self.gradient_surgery == 'Aligned':
            cosine = F.cosine_similarity(agg_sim, agg_smooth, dim=0)
            if cosine < 0:
                grads1 = torch.stack([agg_sim, agg_smooth], dim=0)
                gradients, weight, singulars = ProcrustesSolver.apply(grads1.T.unsqueeze(0), self.scale_mode)
                gradient = gradients[0].sum(-1)
            else:
                gradient = agg_sim
        else:
            raise NotImplementedError
        return gradient

    def __pcgrad(self, main_gradient, auxiliary_gradient):
        cosine = F.cosine_similarity(main_gradient, auxiliary_gradient, dim=0)
        if cosine < 0:
            inner_prod = torch.dot(main_gradient, auxiliary_gradient)
            proj_gradient = inner_prod / torch.linalg.vector_norm(auxiliary_gradient).pow(2) * auxiliary_gradient
            gradient = main_gradient - proj_gradient
        else:
            gradient = main_gradient
        return gradient

    def Cross(self, main_gradient, auxiliary_gradient):
        inner_prod = torch.dot(main_gradient, auxiliary_gradient)
        proj_gradient = inner_prod / torch.linalg.vector_norm(auxiliary_gradient).pow(2) * auxiliary_gradient
        return main_gradient - proj_gradient

    @property
    def device(self):
        return self.model.model_device

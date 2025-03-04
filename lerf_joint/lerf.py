from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type
import random

import numpy as np
import open_clip
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from nerfstudio.viewer.server.viewer_elements import *
from torch.nn import Parameter

from lerf.encoders.image_encoder import BaseImageEncoder
from lerf.lerf_field import LERFField
from lerf.lerf_fieldheadnames import LERFFieldHeadNames
from lerf.lerf_renderers import CLIPRenderer, MeanRenderer


@dataclass
class LERFModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: LERFModel)
    
    clip_loss_weight: float = 10.0
    n_scales: int = 30
    max_scale: float = 1.5 
    """maximum scale used to compute relevancy with"""
    num_lerf_samples: int = 12
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)


class LERFModel(NerfactoModel):
    config: LERFModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.renderer_clip = CLIPRenderer()
        self.renderer_mean = MeanRenderer()

        self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]
        print(self.config.hashgrid_layers)
        self.lerf_field = LERFField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            clip_n_dims=self.image_encoder.embedding_dim,
        )
        self.suppress = True
        self.suppress_train = False
        self.suppress_loss = False
        self.step_counter = 0

        self.use_clip_loss = True
        # populate some viewer logic
        # TODO use the values from this code to select the scale
        # def scale_cb(element):
        #     self.config.n_scales = element.value

        # self.n_scale_slider = ViewerSlider("N Scales", 15, 5, 30, 1, cb_hook=scale_cb)
   
        # def max_cb(element):
        #     self.config.max_scale = element.value

        # self.max_scale_slider = ViewerSlider("Max Scale", 1.5, 0, 5, 0.05, cb_hook=max_cb)

        # def hardcode_scale_cb(element):
        #     self.hardcoded_scale = element.value

        # self.hardcoded_scale_slider = ViewerSlider(
        #     "Hardcoded Scale", 1.0, 0, 5, 0.05, cb_hook=hardcode_scale_cb, disabled=True
        # )

        # def single_scale_cb(element):
        #     self.n_scale_slider.set_disabled(element.value)
        #     self.max_scale_slider.set_disabled(element.value)
        #     self.hardcoded_scale_slider.set_disabled(not element.value)

        # self.single_scale_box = ViewerCheckbox("Single Scale", False, cb_hook=single_scale_cb)

    def get_max_across(self, ray_samples, weights, hashgrid_field, scales_shape, preset_scales=None):
        # TODO smoothen this out
        if preset_scales is not None:
            assert len(preset_scales) == len(self.image_encoder.positives)
            scales_list = torch.tensor(preset_scales)
        else:
            scales_list = torch.linspace(0.0, self.config.max_scale, self.config.n_scales)

        # probably not a good idea bc it's prob going to be a lot of memory
        n_phrases = len(self.image_encoder.positives)
       
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]


        for i, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_output = self.lerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )
            clip_output = self.renderer_clip(embeds=clip_output, weights=weights.detach())

            for j in range(n_phrases):
                if preset_scales is None or j == i:
                    probs = self.image_encoder.get_relevancy(clip_output, j)

                    #probs = self.image_encoder.get_room_relevancy(clip_output, j)

                    pos_prob = probs[..., 0:1]
                    if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                        n_phrases_maxs[j] = scale
                        n_phrases_sims[j] = pos_prob
        
        return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs)

    def get_room_across(self, ray_samples, weights, hashgrid_field, scales_shape, preset_scales=None):
        # TODO smoothen this out
        if preset_scales is not None:
            assert len(preset_scales) == len(self.image_encoder.positives)
            scales_list = torch.tensor(preset_scales)
        else:
            scales_list = torch.linspace(0.0, self.config.max_scale, self.config.n_scales)

        n_phrases = len(self.image_encoder.positives)
       
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]

        for i, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_output = self.lerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )
            clip_output = self.renderer_clip(embeds=clip_output, weights=weights.detach())

            j = 0
            if preset_scales is None or j == i:
                    #probs = self.image_encoder.get_relevancy(clip_output, j)

                    probs = self.image_encoder.get_room_relevancy(clip_output, j)

                    pos_prob = probs[..., 0:1] - probs[..., 1:]
                    #print(pos_prob)
                    if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                        n_phrases_maxs[j] = scale
                        n_phrases_sims[j] = pos_prob
        
        return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs)

        """
        room_max = 0.0
        room_sim = torch.zeros((1,1))
        start = True
        for i, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_output = self.lerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )
            clip_output = self.renderer_clip(embeds=clip_output, weights=weights.detach())
 #print(outputs['rgb'][:, 0]*over_thresh)
                print('here')
            if preset_scales is None:
                
                probs = self.image_encoder.get_room_relevancy(clip_output, 0)
                print(probs)
                pos_prob = probs[..., 0:1]
                if start or pos_prob.max() > room_sim.max():
                        room_max = scale
                        room_sim = pos_prob
                        start = False
        
        print(room_sim)
        print(room_max)
        return room_sim.unsqueeze(dim=0), torch.Tensor([room_max])
        """

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)
        
        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples)
        lerf_weights, best_ids = torch.topk(weights, self.config.num_lerf_samples, dim=-2, sorted=False)
        #print(torch.cat((ray_samples,nerfacto_field_outputs['embeds'])).shape)
        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))
        
        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        lerf_samples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)
        
        
        if self.training:
            clip_scales = ray_bundle.metadata["clip_scales"]
            clip_scales = clip_scales[..., None]
            dist = lerf_samples.spacing_to_euclidean_fn(lerf_samples.spacing_starts.squeeze(-1)).unsqueeze(-1)
            clip_scales = clip_scales * ray_bundle.metadata["width"] * (1 / ray_bundle.metadata["fx"]) * dist
        else:
            clip_scales = torch.ones_like(lerf_samples.spacing_starts, device=self.device)

        override_scales = (
            None if "override_scales" not in ray_bundle.metadata else ray_bundle.metadata["override_scales"]
        )
        weights_list.append(weights)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        lerf_field_outputs = self.lerf_field.get_outputs(lerf_samples, clip_scales,  gather_fn(nerfacto_field_outputs['embeds']))

        if self.training:
            outputs["clip"] = self.renderer_clip(
                embeds=lerf_field_outputs[LERFFieldHeadNames.CLIP], weights=lerf_weights.detach()
            )
            outputs["dino"] = self.renderer_mean(
                embeds=lerf_field_outputs[LERFFieldHeadNames.DINO], weights=lerf_weights.detach()
            )

        if self.suppress_train and self.training:
            with torch.no_grad():
                if random.randint(0, 10) > 0:
                    #print('suppressing during training...')
                    self.step_counter += 1
                    if self.step_counter >10000:
                        print(outputs_clip)
                        max_across, best_scales = self.get_room_across(
                                    lerf_samples,
                                    lerf_weights,
                                    lerf_field_outputs[LERFFieldHeadNames.HASHGRID],
                                    clip_scales.shape,
                                    preset_scales=override_scales,
                                )
                        threshold = 0.0
                        #outputs["raw_relevancy"] = max_across  # N x B x 1
                        #outputs["best_scales"] = best_scales.to(self.device)  # N
                        #outputs["depth"] = torch.where((outputs["depth"].squeeze() * (outputs["raw_relevancy"] >= threshold).long().squeeze()) == 0, 0, (outputs["depth"].squeeze() * (outputs["raw_relevancy"] >= threshold).long().squeeze())).unsqueeze(dim=1)  
                        outputs['depth'] = (outputs['depth'].squeeze()*(max_across >= threshold).long().squeeze()).unsqueeze(dim=1)
                        

                        #outputs["depth"] = torch.where((outputs["depth"].squeeze() * (outputs["raw_relevancy"] >= threshold).long().squeeze()) == 0, 100000, (outputs["depth"].squeeze() * (outputs["raw_relevancy"] >= threshold).long().squeeze())).unsqueeze(dim=1)  
                        #outputs['rgb'][:, 0] = outputs['rgb'][:, 0]*(max_across >= threshold).long().squeeze()
                        #outputs['rgb'][:, 1] = outputs['rgb'][:, 1]*(max_across >= threshold).long().squeeze()
                        #outputs['rgb'][:, 2] = outputs['rgb'][:, 2]*(max_across >= threshold).long().squeeze()
                        del max_across, best_scales

        if self.suppress_loss and self.training:
            
                if random.randint(0, 10) > 0:
                    #print('suppressing during training...')
                    self.step_counter += 1
                    if self.step_counter >1:
                        max_across, best_scales = self.get_room_across(
                                    lerf_samples,
                                    lerf_weights,
                                    lerf_field_outputs[LERFFieldHeadNames.HASHGRID],
                                    clip_scales.shape,
                                    preset_scales=override_scales,
                                )
                        threshold = 0.0
                        
                        outputs["max_across"] = max_across

                        del max_across, best_scales
                    else:
                        outputs["suppression_loss"] = 0

        if not self.training:
            with torch.no_grad():
                
                if self.suppress: 
                    
                    max_across, best_scales = self.get_room_across(
                        lerf_samples,
                        lerf_weights,
                        gather_fn(nerfacto_field_outputs['embeds']),
                        clip_scales.shape,
                        preset_scales=override_scales,
                    )
                    threshold = 0.0
                    outputs["raw_relevancy"] = max_across  # N x B x 1
                    outputs["best_scales"] = best_scales.to(self.device)  # N
                    #print(outputs)
                    #print(outputs["depth"])
                    #print((outputs["raw_relevancy"] >= threshold).long().squeeze())
                    #print(torch.where((outputs["raw_relevancy"] >= threshold).long().squeeze() == 0, -100, (outputs["raw_relevancy"] >= threshold).long().squeeze()))
                    #print((outputs["raw_relevancy"] >= threshold).long().squeeze())
                    #outputs["accumulation"] = (outputs["accumulation"].squeeze() * (outputs["raw_relevancy"] >= threshold).long().squeeze()).unsqueeze(dim=1)
                    outputs["depth"] = torch.where((outputs["depth"].squeeze() * (outputs["raw_relevancy"] >= threshold).long().squeeze()) == 0, 100000, (outputs["depth"].squeeze() * (outputs["raw_relevancy"] >= threshold).long().squeeze())).unsqueeze(dim=1)
                    
                    #outputs["prop_depth_0"] = (outputs["prop_depth_0"].squeeze() *(outputs["raw_relevancy"] >= threshold).long().squeeze()).unsqueeze(dim=1)
                    #outputs["prop_depth_1"] = (outputs["prop_depth_1"].squeeze() *(outputs["raw_relevancy"] >= threshold).long().squeeze()).unsqueeze(dim=1)

                    #print(outputs["accumulation"])
                    #outputs['rgb'][:, 0] = outputs['rgb'][:, 0]*(outputs["raw_relevancy"] >= threshold).long().squeeze()
                    #outputs['rgb'][:, 1] = outputs['rgb'][:, 1]*(outputs["raw_relevancy"] >= threshold).long().squeeze()
                    #outputs['rgb'][:, 2] = outputs['rgb'][:, 2]*(outputs["raw_relevancy"] >= threshold).long().squeeze()
                
                   
                else:
                    max_across, best_scales = self.get_max_across(
                        lerf_samples,
                        lerf_weights,
                        lerf_field_outputs[LERFFieldHeadNames.HASHGRID],
                        clip_scales.shape,
                        preset_scales=override_scales,
                    )
                    outputs["raw_relevancy"] = max_across  # N x B x 1
                    outputs["best_scales"] = best_scales.to(self.device)  # N
               
        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        LERF overrides this from base_model since we need to compute the max_across relevancy in multiple batches,
        which are not independent since they need to use the same scale
        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        # TODO(justin) implement max across behavior
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)  # dict from name:list of outputs (1 per bundle)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            # take the best scale for each query across each ray bundle
            if i == 0:
                best_scales = outputs["best_scales"]
                best_relevancies = [m.max() for m in outputs["raw_relevancy"]]
            else:
                for phrase_i in range(outputs["best_scales"].shape[0]):
                    m = outputs["raw_relevancy"][phrase_i, ...].max()
                    if m > best_relevancies[phrase_i]:
                        best_scales[phrase_i] = outputs["best_scales"][phrase_i]
                        best_relevancies[phrase_i] = m
        # re-render the max_across outputs using the best scales across all batches
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            ray_bundle.metadata["override_scales"] = best_scales
            outputs = self.forward(ray_bundle=ray_bundle)
            # standard nerfstudio concatting
            for output_name, output in outputs.items():  # type: ignore
                if output_name == "best_scales":
                    continue
                if output_name == "raw_relevancy":
                    for r_id in range(output.shape[0]):
                        outputs_lists[f"relevancy_{r_id}"].append(output[r_id, ...])
                else:
                    outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        for i in range(len(self.image_encoder.positives)):
            p_i = torch.clip(outputs[f"relevancy_{i}"] - 0.5, 0, 1)
            outputs[f"composited_{i}"] = apply_colormap(p_i / (p_i.max() + 1e-6), ColormapOptions("turbo"))
            mask = (outputs["relevancy_0"] < 0.5).squeeze()
            outputs[f"composited_{i}"][mask, :] = outputs["rgb"][mask, :]
        return outputs

    def _get_outputs_nerfacto(self, ray_samples: RaySamples):
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        return field_outputs, outputs, weights

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training and self.use_clip_loss:
            unreduced_clip = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
                outputs["clip"], batch["clip"], delta=1.25, reduction="none"
            )
            
            loss_dict["clip_loss"] = unreduced_clip.sum(dim=-1).nanmean()
            #unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
            #loss_dict["dino_loss"] = unreduced_dino.sum(dim=-1).nanmean()
            if self.suppress_loss:
                try:
                    ones = torch.ones_like(outputs["max_across"].squeeze())
                    loss_dict["suppression_loss"] = 2 * torch.nn.functional.binary_cross_entropy_with_logits(outputs["max_across"].squeeze().requires_grad_(), ones)
                    
                except:
                    pass
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["lerf"] = list(self.lerf_field.parameters())
        return param_groups

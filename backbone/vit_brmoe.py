import copy
import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
from einops import rearrange, einsum
from timm.models.vision_transformer import DropPath, PatchEmbed


class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 adapter_layernorm_option="in",
                 non_linear_func=nn.ReLU,
                ):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = non_linear_func()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scale = nn.Parameter(torch.zeros(self.n_embd), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, add_residual=False, residual=None):

        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)

        down = self.dropout(down)
        up = self.up_proj(down)
        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

    

class MoEAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dim = config.d_model

        self.expert_dim = config.get('expert_dim', 16)
        self.adapter_dim = config.get('adapter_dim', 64)
        self.topk_routers = config.get('topk_routers', 2)
        self.topk = config.get('topk_experts', 3)

        self.norm = nn.ModuleList()
        self.adapter = nn.ModuleList()
        self.router = nn.ModuleList()

        self.shared_adapter = Adapter(config, bottleneck=self.adapter_dim, dropout=0.1)

        self.latent_head = nn.ModuleList()
        self._update_adapters()
        # self.router[0].requires_grad_(False)
    

    def _init_adapter(self):

        adapter = Adapter(self.config, bottleneck=self.expert_dim, dropout=0.1)
        adapter = copy.deepcopy(adapter)

        if len(self.adapter) > 1:
            # initialize from the previous adapters instead of random initialization for more stable training
            params = {}
            for i in range(len(self.adapter)-1):
                for name, param in self.adapter[i].named_parameters():
                    params[name] = params.get(name, []) + [param.data.clone()]

            for name, param_list in params.items():
                mean_param = torch.mean(torch.stack(param_list, dim=0), dim=0)
                params[name] = mean_param

            for name, param in adapter.named_parameters():
                param.data = params[name].data.clone()

        return adapter


    def _update_adapters(self, num_classes=None):

        adapter = self._init_adapter()
        self.adapter.append(adapter.to(self.config._device))
        self.router.append(nn.Linear(self.dim, len(self.adapter)).to(self.config._device))
        nn.init.trunc_normal_(self.router[-1].weight, std=0.02)
        nn.init.zeros_(self.router[-1].bias)

        self._freeze_params()

        self.shared_adapter.requires_grad_(True)

        num_classes = self.config.init_cls if num_classes is None else num_classes

        latent_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, num_classes),
        )

        self.latent_head.append(latent_head).to(self.config._device)

        nn.init.trunc_normal_(self.latent_head[-1][0].weight, std=0.02)
        nn.init.zeros_(self.latent_head[-1][0].bias)
        
        self.adapter[-1].requires_grad_(True)
        self.router[-1].requires_grad_(True)
        self.latent_head[-1].requires_grad_(True)


    def _freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False


    def _load_shared_adapter(self, use_ema=False):
        
        self.shared_adapter.requires_grad_(True)

        if len(self.adapter) > 1:
            self._shared_adapter = copy.deepcopy(self.shared_adapter)
            self._shared_adapter.requires_grad_(False)

            if use_ema:
                self.ema_adapter = copy.deepcopy(self._shared_adapter)
                self.ema_adapter.requires_grad_(False)
                del self._shared_adapter


    def _get_shared_adapter(self, ema_decay=0.999):
        
        shared_adapter = self.shared_adapter

        if self.training:
            if hasattr(self, 'ema_adapter'):
                for param, ema_param in zip(shared_adapter.parameters(), 
                                            self.ema_adapter.parameters()):
                    param.data = ema_decay * ema_param.data + (1 - ema_decay) * param.data

        else:
            # ema adapter is not needed during inference
            if hasattr(self, 'ema_adapter'):
                del self.ema_adapter
            elif hasattr(self, '_shared_adapter'):
                del self._shared_adapter

        return shared_adapter
    

    def semi_dynamic_forward(self, x, task_id=0):

        outs = []
        latents = []
        logits = []

        max_experts = task_id + 1

        for i in range(max_experts):

            out = self.adapter[i](x)
            outs.append(out)

            cls_token = out[:, 0, :]  # (B, C)

            latent = self.latent_head[i](cls_token)
            latents.append(latent)

            score = self.router[i](cls_token)

            if score.shape[-1] < max_experts:
                pad = torch.ones(score.size(0), max_experts-score.size(-1), device=x.device) * float('-inf')
                score = torch.cat([score, pad], dim=-1)

            logits.append(score)

        if task_id == 0:
            return (outs[0], latents[0])

        outs = torch.stack(outs, dim=1)  # (B, E, N, C)
        
        all_logits = torch.stack(logits, dim=0)  # (T, B, E)

        if latents[0].shape[-1] != latents[1].shape[-1]:
            # init cls != incremental cls
            latents_t0 = latents[0].unsqueeze(0)
            if len(latents) > 2:
                pre_cls = latents[1:task_id] # (T-1, B, Cls)
                pre_cls = torch.stack(pre_cls, dim=0)
            else:
                pre_cls = None
        else:
            latents_t0 = None
            pre_cls = latents[:task_id] # (T-1, B, Cls)
            pre_cls = torch.stack(pre_cls, dim=0)

        cls = latents[task_id] # (B, Cls)

        pre_logits = all_logits[:task_id] # (T-1, B, E)
        cur_logits = all_logits[task_id] # (B, E)

        if self.topk_routers == 1:

            logits = cur_logits # (B, E)

            if (self.topk is not None) and (self.topk < max_experts):
                topk_idx = logits.topk(self.topk, dim=-1)[1]  # (B, topk)
                mask = torch.zeros_like(logits).bool()  # (B, E)
                mask.scatter_(-1, topk_idx, True)
                logits = logits.masked_fill(~mask, -100)

            logits = torch.softmax(logits, dim=-1) # (B, E)
            x = einsum(logits, outs, 'b e, b e n c -> b n c') # (B, N, C)

            return (x, cls)

        else:

            # use topk-related routers to conduct inference
            if latents_t0 is not None:
                if len(latents) <= 2:
                    # pre_cls = latents_t0
                    probs = torch.softmax(latents_t0, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                else:
                    probs_t0 = torch.softmax(latents_t0, dim=-1)
                    probs_pre = torch.softmax(pre_cls, dim=-1)
                    entropy_t0 = -torch.sum(probs_t0 * torch.log(probs_t0 + 1e-10), dim=-1)
                    entropy_pre = -torch.sum(probs_pre * torch.log(probs_pre + 1e-10), dim=-1)
                    entropy = torch.cat([entropy_t0, entropy_pre], dim=0)
            else:
                probs = torch.softmax(pre_cls, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

            topk_routers = self.topk_routers - 1 # exclude the main router

            if topk_routers == 1:
                entropy_idx = torch.argmin(entropy, dim=0)
                logits = pre_logits[entropy_idx, torch.arange(len(entropy_idx))]
                logits = torch.stack([logits, cur_logits], dim=0) # (2, B, E)

            elif topk_routers < max_experts - 1:

                entropy_idx = entropy.topk(topk_routers, dim=0, largest=False)[1] # (B, topk_routers)
                entropy_idx = entropy_idx.unsqueeze(-1).expand(-1, -1, pre_logits.size(-1))
                pre_logits = torch.gather(pre_logits, dim=0, index=entropy_idx) # (topk_routers, B, E)
                logits = torch.cat([pre_logits, cur_logits.unsqueeze(0)], dim=0) # (T, B, E)

            else:
                topk_routers = max_experts
                logits = all_logits # (T, B, E)

            if (self.topk is not None) and (self.topk < max_experts):
                topk_idx = logits.topk(self.topk, dim=-1)[1]  # (T, B, topk)
                mask = torch.zeros_like(logits).bool()  # (T, B, topk)
                mask.scatter_(-1, topk_idx, True)
                logits = logits.masked_fill(~mask, -100)

            logits = torch.softmax(logits, dim=-1)  # (topk_routers, B, E)
            r = logits.size(0)
            x = outs.unsqueeze(1).expand(-1, r, -1, -1, -1)
            x = einsum(logits, x, 'r b e, b r e n c -> b n c')  # (B, N, C)

        return (x, cls)
        

    def full_dynamic_forward(self, x, task_id=None):

        task_id = None # task_id is unknown in this forward pass

        if len(self.adapter) == 1:
            x = self.adapter[0](x)
            return x

        else:

            logits = []
            outs = []
            latents = []

            max_experts = len(self.adapter)

            for i in range(max_experts):

                out = self.adapter[i](x)
                cls_token = out[:, 0, :]  # (B, C)
                latent = self.latent_head[i](cls_token)

                outs.append(out)
                latents.append(latent)

                score = self.router[i](cls_token)
                
                if score.shape[-1] < max_experts:
                    pad = torch.ones(score.size(0), max_experts-score.size(-1), device=x.device) * float('-inf')
                    score = torch.cat([score, pad], dim=-1)
                logits.append(score) # (B, 1)

            outs = torch.stack(outs, dim=1)  # (B, E, N, C)
            logits = torch.stack(logits, dim=0)  # (T, B, E)

            if latents[0].shape[-1] != latents[1].shape[-1]:
                # init cls != incremental cls
                latents_t0 = latents[0].unsqueeze(0)
                latents_rest = torch.stack(latents[1:], dim=0)
                probs_t0 = torch.softmax(latents_t0, dim=-1)
                probs_rest = torch.softmax(latents_rest, dim=-1)
                entropy_t0 = -torch.sum(probs_t0 * torch.log(probs_t0 + 1e-10), dim=-1)
                entropy_rest = -torch.sum(probs_rest * torch.log(probs_rest + 1e-10), dim=-1)
                entropy = torch.cat([entropy_t0, entropy_rest], dim=0)
            else:
                latents = torch.stack(latents, dim=0) # (T, B, Cls)
                probs = torch.softmax(latents, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

            if self.topk_routers == 1:
                entropy_idx = torch.argmin(entropy, dim=0)
                logits = logits[entropy_idx, torch.arange(len(entropy_idx))]

                if (self.topk is not None) and (self.topk < len(self.adapter)):
                    topk_idx = logits.topk(self.topk, dim=-1)[1]  # (B, topk)
                    mask = torch.zeros_like(logits).bool()  # (B, E)
                    mask.scatter_(-1, topk_idx, True)
                    logits = logits.masked_fill(~mask, -100)

                logits = torch.softmax(logits, dim=-1)  # (B, E)
                x = einsum(logits, outs, 'b e, b e n c -> b n c')  # (B, N, C)

            else:
                # use topk-related routers to conduct inference
                if self.topk_routers < max_experts:
                    topk_routers = self.topk_routers
                    entropy_idx = entropy.topk(topk_routers, dim=0, largest=False)[1] # (B, topk_routers)
                    entropy_idx = entropy_idx.unsqueeze(-1).expand(-1, -1, logits.size(-1))
                    logits = torch.gather(logits, dim=0, index=entropy_idx) # (topk_routers, B, E)
                else:
                    topk_routers = max_experts

                if (self.topk is not None) and (self.topk < len(self.adapter)):
                    topk_idx = logits.topk(self.topk, dim=-1)[1]  # (B, topk_routers, topk)
                    mask = torch.zeros_like(logits).bool()  # (B, topk_routers, E)
                    mask.scatter_(-1, topk_idx, True)
                    logits = logits.masked_fill(~mask, -100)

                logits = torch.softmax(logits, dim=-1)  # (topk_routers, B, E)
                r = logits.size(0)
                x = outs.unsqueeze(1).expand(-1, r, -1, -1, -1)
                x = einsum(logits, x, 'r b e, b r e n c -> b n c')  # (B, N, C)

            return x


    def forward(self, x, add_residual=True, residual=None, task_id=0, with_task_id=False):

        residual = x if residual is None else residual

        adapter = self._get_shared_adapter()
        s = adapter(x)

        if with_task_id:
            (x, cls) = self.semi_dynamic_forward(x, task_id)
        else:
            x = self.full_dynamic_forward(x, task_id)
            cls = None
        
        x = x + s

        if add_residual:
            x = x + residual

        return (x, cls)
        
        

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x, adapt=None, task_id=0, with_task_id=False):

        x = x + self.drop_path(self.attn(self.norm1(x)))

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        cls = None

        if adapt is not None:
            if self.config.ffn_option == 'sequential':
                x, cls = adapt(x, add_residual=True, task_id=task_id, with_task_id=with_task_id)
            elif self.config.ffn_option == 'parallel':
                adapt_x, cls = adapt(residual, add_residual=False, task_id=task_id, with_task_id=with_task_id)
                x = x + adapt_x
        x = residual + x

        return (x, cls)


class VisionTransformer(nn.Module):
    """ 
    Vision Transformer with support for global average pooling
    """

    def __init__(self, 
                 global_pool=False, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768,
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 representation_size=None, 
                 distilled=False,
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 embed_layer=PatchEmbed, 
                 norm_layer=None,
                 act_layer=None, 
                 weight_init='', 
                 tuning_config=None, 
                 r=16):
        super().__init__()

        print("I'm using ViT with adapters.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.adapter_idx = tuple(range(depth))

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        ######### MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm


        ######### Initialize adapters #########
        self.adapter = self.init_adapters(layer_idx=self.adapter_idx)


    def init_adapters(self, layer_idx):

        adapters = nn.ModuleDict()

        for idx in layer_idx:
            adapter = MoEAdapter(self.tuning_config)
            adapter.requires_grad_(True)
            adapters.update(
                {f'adapter_layer{idx}': adapter}
            )

        return adapters
    
    
    def update_adapters(self, num_classes=None):
         for module in self.adapter.values():
             module._update_adapters(num_classes)


    def train_shared_adapter(self, use_ema=True):
        for module in self.adapter.values():
            module._load_shared_adapter(use_ema)


    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


    def forward_features(self, x, adapter_id, with_task_id=False):

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        latents = []
        if adapter_id == -1:
            x = self.blocks(x)
        else:
            for layer_idx, blk in enumerate(self.blocks):
                adapt = None
                if layer_idx in self.adapter_idx:
                    adapt = self.adapter[f'adapter_layer{layer_idx}']
                x, cls = blk(x, adapt, task_id=adapter_id, with_task_id=with_task_id)
                if cls is not None:
                    latents.append(cls)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        res = dict()
        res['x'] = outcome
        res['pre_logits'] = outcome
        res['features'] = outcome
        res['latents'] = latents

        return res

    def forward_head(self, res):
        x = res['x']
        res['logits'] = self.head(x)
        return res

    def forward(self, x, adapter_id=-1, with_task_id=False, fc_only=False):

        if fc_only:
            res = dict()
            res['logits'] = self.head(x)
            return res

        res = self.forward_features(x, adapter_id, with_task_id)
        res = self.forward_head(res)

        return res


def vit_base_patch16_224_brmoe(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=kwargs['num_classes'])
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768 * 2]
            v_weight = qkv_weight[768 * 2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768 * 2]
            v_bias = qkv_bias[768 * 2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)

    # freeze all but the adapter and head
    for name, p in model.named_parameters():
        if 'head' in name or 'adapter' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def vit_base_patch16_224_in21k_brmoe(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True,
                                         num_classes=kwargs['num_classes'])
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768 * 2]
            v_weight = qkv_weight[768 * 2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768 * 2]
            v_bias = qkv_bias[768 * 2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)

    # freeze all but the adapter and head
    for name, p in model.named_parameters():
        if 'head' in name or 'adapter' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model




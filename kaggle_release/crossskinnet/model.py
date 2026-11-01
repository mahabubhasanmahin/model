import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class CrossSkinNet(nn.Module):
    def __init__(self, cfg, schema, hf_config=None):
        super().__init__()
        self.kind = cfg['backbone']
        self.mode = cfg.get('mode', 'gated')
        if self.mode not in ('gated', 'image_only', 'metadata_only', 'naive'):
            raise ValueError('mode: gated | image_only | metadata_only | naive')
        self.vision = None
        if self.mode == 'metadata_only':
            dim = cfg.get('metadata_only_dim', 256)
        elif self.kind == 'tiny':
            self.vision = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                        nn.AdaptiveAvgPool2d(1), nn.Flatten())
            dim = 16
        else:
            expected = {'dinov3': 'dinov3_vit', 'dinov2': 'dinov2',
                        'efficientnet': 'efficientnet', 'swin': 'swin'}
            if self.kind not in expected:
                raise ValueError('Unsupported backbone')
            if hf_config is None:
                self.vision = AutoModel.from_pretrained(cfg['path'])
            else:
                args = dict(hf_config)
                self.vision = AutoModel.from_config(AutoConfig.for_model(args.pop('model_type'), **args))
            if self.vision.config.model_type != expected[self.kind]:
                raise ValueError('backbone and checkpoint architecture differ')
            dim = self.vision.config.hidden_dim if self.kind == 'efficientnet' else self.vision.config.hidden_size
        self.num_numeric = len(schema['numeric'])
        self.fields = nn.ModuleList([nn.Linear(1, cfg.get('token_dim', 32)) for _ in schema['numeric']] +
                                    [nn.Embedding(len(f['vocab'])+1, cfg.get('token_dim',32)) for f in schema['categorical']])
        count, width = len(self.fields), cfg.get('token_dim',32)
        if not count and self.mode == 'metadata_only':
            raise ValueError('metadata_only requires metadata features')
        self.missing = nn.Parameter(torch.randn(count, width)*.02)
        # Naive baseline uses zero-imputed values without missing tokens/mask input.
        if count:
            meta_in = count*width + (0 if self.mode == 'naive' else count)
            self.metadata = nn.Sequential(nn.Linear(meta_in,dim),nn.ReLU(),nn.LayerNorm(dim),nn.Dropout(.3))
            self.fusion = nn.Linear(2*dim,dim)
            self.gate = nn.Linear(2*dim,dim)
        self.head = nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Dropout(.3),nn.Linear(dim,len(schema['classes'])))

    def image_features(self, pixels):
        if self.kind == 'tiny':
            return self.vision(pixels)
        out = self.vision(pixel_values=pixels)
        return out.last_hidden_state[:,0] if self.kind in ('dinov2','dinov3') else out.pooler_output.flatten(1)

    def forward(self, pixels, values, mask):
        image = self.image_features(pixels) if self.vision is not None else None
        if self.mode == 'image_only' or not self.fields:
            return self.head(image)
        mask = mask.bool()
        clean = torch.where(mask, values, torch.zeros_like(values))
        tokens=[]
        for i,layer in enumerate(self.fields):
            token=layer(clean[:,i:i+1] if i<self.num_numeric else clean[:,i].long())
            token=torch.where(mask[:,i:i+1],token,torch.zeros_like(token) if self.mode=='naive' else self.missing[i])
            tokens.append(token)
        flat=torch.stack(tokens,1).flatten(1)
        meta=self.metadata(flat if self.mode=='naive' else torch.cat([flat,mask.float()],1))
        if self.mode=='metadata_only':
            return self.head(meta)
        joint=torch.cat([image,meta],1)
        if self.mode=='naive':
            return self.head(self.fusion(joint))
        fused=image+self.fusion(joint)*self.gate(joint).sigmoid()
        return self.head(torch.where(mask.any(1,keepdim=True),fused,image))

    def configure_phase(self, blocks=0):
        if self.vision is None:
            return
        self.vision.requires_grad_(False)
        if not blocks:
            return
        if self.kind=='tiny':
            self.vision.requires_grad_(True)
            return
        if self.kind=='dinov3':
            layers=list(self.vision.layer)
        elif self.kind=='dinov2':
            layers=list(self.vision.encoder.layer)
        elif self.kind=='swin':
            layers=[b for stage in self.vision.encoder.layers for b in stage.blocks]
        else:
            layers=list(self.vision.encoder.blocks)
        if not isinstance(blocks,int) or not 1<=blocks<=len(layers):
            raise ValueError(f'unfreeze_blocks must be between 1 and {len(layers)}')
        for block in layers[-blocks:]:
            block.requires_grad_(True)

    def architecture(self):
        return self.vision.config.to_dict() if self.vision is not None and self.kind!='tiny' else None

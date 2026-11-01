"""Optional qualitative explanations, not evidence of clinical correctness."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from .data import encode,read_csv,loader,json_save
from .engine import load_model


def cam_target(model):
    if model.vision is None:
        raise ValueError('Metadata-only model has no image Grad-CAM')
    if model.kind=='tiny':
        return model.vision[0]
    if model.kind=='dinov3':
        block=model.vision.layer[-1]
    elif model.kind=='dinov2':
        block=model.vision.encoder.layer[-1]
    elif model.kind=='swin':
        block=model.vision.encoder.layers[-1].blocks[-1]
    else:
        return list(m for m in model.vision.encoder.modules() if isinstance(m,nn.Conv2d))[-1]
    # Hook before the final attention operation, not the post-attention CLS norm.
    return next(m for m in block.modules() if isinstance(m,nn.LayerNorm))


def gradcam(model,pixels,values,mask,class_index=None):
    captured={}
    def hook(module,args,output):
        captured['activation']=output
    handle=cam_target(model).register_forward_hook(hook)
    try:
        # Even a frozen backbone needs differentiable activations for XAI.
        pixels=pixels.detach().requires_grad_(True)
        logits=model(pixels,values,mask)
        target=int(logits.argmax(1)[0]) if class_index is None else class_index
        if not 0<=target<logits.shape[1]:
            raise ValueError('Invalid class index')
        activation=captured['activation']
        gradients=torch.autograd.grad(logits[0,target],activation,allow_unused=True)[0]
        if gradients is None:
            raise RuntimeError('Chosen class does not depend on this image activation')
        if activation.ndim==3:
            if model.kind in ('dinov2','dinov3'):
                patch=model.vision.config.patch_size
                if not isinstance(patch,int):
                    patch=patch[0]
                height,width=pixels.shape[-2]//patch,pixels.shape[-1]//patch
                count=height*width
                activation=activation[:,-count:]
                gradients=gradients[:,-count:]
            else:
                count=activation.shape[1];height=width=math.isqrt(count)
                if height*width!=count:
                    raise ValueError('This Grad-CAM adapter requires a square Swin token grid')
            activation=activation.transpose(1,2).reshape(1,-1,height,width)
            gradients=gradients.transpose(1,2).reshape(1,-1,height,width)
        elif activation.ndim!=4:
            raise ValueError('Unsupported activation shape')
        heat=(gradients.mean((2,3),keepdim=True)*activation).sum(1).relu()
        if not torch.isfinite(heat).all() or float(heat.max())<=1e-12:
            raise RuntimeError('Degenerate/zero Grad-CAM; do not present it as a meaningful explanation')
        heat=(heat-heat.min())/(heat.max()-heat.min()+1e-8)
        return heat[0].detach().cpu().numpy(),target
    finally:
        handle.remove()


def explain(checkpoint,image,metadata,device,output_dir,class_index=None,with_shap=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    model,processor,saved=load_model(checkpoint,device)
    with Image.open(image) as source:
        raw=source.convert('RGB')
    if processor:
        pixels=processor(images=raw,return_tensors='pt')['pixel_values'].to(device)
    else:
        n=saved['config']['model'].get('image_size',32)
        pixels=torch.from_numpy(np.array(raw.resize((n,n)),dtype=np.float32)).permute(2,0,1).unsqueeze(0).to(device)/255.
    values,mask=encode(metadata,saved['schema']);values=values.unsqueeze(0).to(device);mask=mask.unsqueeze(0).to(device)
    heat,target=gradcam(model,pixels,values,mask,class_index)
    # Display in processed-image coordinates: processor may crop the raw image.
    if processor:
        mean=np.array(getattr(processor,'image_mean',[0,0,0]));std=np.array(getattr(processor,'image_std',[1,1,1]))
        shown=np.clip(pixels[0].detach().cpu().permute(1,2,0).numpy()*std+mean,0,1)
    else:
        shown=pixels[0].cpu().permute(1,2,0).numpy()
    heat=np.array(Image.fromarray((heat*255).astype('uint8')).resize((shown.shape[1],shown.shape[0]),Image.Resampling.BILINEAR))/255.
    fig,ax=plt.subplots();ax.imshow(shown);ax.imshow(heat,cmap='jet',alpha=.4);ax.axis('off')
    ax.set_title(saved['schema']['classes'][target]);fig.tight_layout();fig.savefig(out/'gradcam.png',dpi=180);plt.close(fig)
    report={'class':saved['schema']['classes'][target],
            'gradcam_coordinates':'model-processed image, not uncropped raw image',
            'warning':'Qualitative association only; heatmaps do not prove lesion-only reasoning or clinical safety.'}
    if with_shap:
        try:
            import shap
        except ImportError as e:
            raise ImportError('Install requirements-xai.txt for metadata SHAP') from e
        schema=saved['schema'];names=[f['name'] for f in schema['numeric']+schema['categorical']]
        if not names:
            raise ValueError('No metadata features for SHAP')
        frame=read_csv(Path(checkpoint).parent/'train_split.csv').sample(n=min(20,len(read_csv(Path(checkpoint).parent/'train_split.csv'))),random_state=42)
        background=np.stack([encode(row,schema)[0].numpy() for row in frame.to_dict('records')])
        def probability(matrix):
            with torch.no_grad():
                v=torch.tensor(matrix,dtype=torch.float32,device=device)
                # Mask is fixed to this patient's availability, including missing values.
                output=model(pixels.expand(len(v),-1,-1,-1),v,mask.expand(len(v),-1))
                return output.softmax(1)[:,target].cpu().numpy()
        explainer=shap.KernelExplainer(probability,background)
        attribution=np.asarray(explainer.shap_values(values.cpu().numpy(),nsamples=100)).reshape(-1)
        pd.DataFrame({'feature':names,'shap_value':attribution,'observed':mask[0].cpu().tolist()}).to_csv(out/'metadata_shap.csv',index=False)
        fig,ax=plt.subplots();ax.barh(names,attribution);ax.set_xlabel('SHAP contribution to selected-class probability')
        fig.tight_layout();fig.savefig(out/'metadata_shap.png',dpi=180);plt.close(fig)
        report['shap_scope']='Single image, fixed availability mask; train-derived encoded background, not causal effects or availability attribution.'
    json_save(out/'explanation.json',report)
    return report

import copy
import json
import math
import os
import random
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix
from transformers import AutoImageProcessor
from .model import CrossSkinNet
from .data import prepare_splits,fit_schema,loader,json_save,read_csv,encode


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def processor_for(config,saved=None):
    if config['model']['backbone']=='tiny':
        return None
    # DINOv3 requires the fast torchvision processor in Transformers 4.57.
    fast=config['model'].get('use_fast',config['model']['backbone']=='dinov3')
    try:
        return AutoImageProcessor.from_pretrained(saved or config['model']['path'],
                                                  use_fast=fast,local_files_only=saved is not None)
    except (ImportError,ValueError,RuntimeError) as e:
        raise RuntimeError('Image processor failed. Check preprocessor_config.json and matching torch/torchvision. '
                           'DINOv3 needs use_fast: true. Do not replace checkpoint normalization arbitrarily.') from e


def save_atomic(path,state):
    path=Path(path); temporary=path.with_suffix('.tmp')
    torch.save(state,temporary); os.replace(temporary,path)


def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n=state['numpy']; np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        if len(state['cuda'])!=torch.cuda.device_count():
            raise ValueError('Exact resume requires the same visible GPU count')
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def metric_report(truth,predicted,classes):
    ids=list(range(len(classes)))
    return {'accuracy':float(accuracy_score(truth,predicted)),
            'balanced_accuracy':float(balanced_accuracy_score(truth,predicted)),
            'macro_f1':float(f1_score(truth,predicted,labels=ids,average='macro',zero_division=0)),
            'per_class':classification_report(truth,predicted,labels=ids,target_names=classes,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(truth,predicted,labels=ids).tolist()}


@torch.no_grad()
def evaluate_batches(model,batches,device,classes,drop=0.,seed=42):
    model.eval(); truths=[]; probs=[]
    rng=torch.Generator().manual_seed(seed)
    for pixels,values,mask,labels in batches:
        mask=mask&(torch.rand(mask.shape,generator=rng)>=drop)
        logits=model(pixels.to(device),values.to(device),mask.to(device))
        probs.extend(logits.softmax(-1).cpu().tolist()); truths.extend(labels.tolist())
    probabilities=np.array(probs)
    return metric_report(truths,probabilities.argmax(1),classes),np.array(truths),probabilities


def parameter_groups(model,phase):
    vision_ids={id(p) for p in model.vision.parameters()} if model.vision is not None else set()
    head=[p for p in model.parameters() if p.requires_grad and id(p) not in vision_ids]
    backbone=[p for p in model.parameters() if p.requires_grad and id(p) in vision_ids]
    groups=[{'params':head,'lr':float(phase['head_lr'])}]
    if backbone:
        groups.append({'params':backbone,'lr':float(phase['backbone_lr'])})
    return groups


def make_scheduler(optimizer,total,warmup_fraction):
    warmup=int(total*warmup_fraction)
    def multiplier(step):
        if warmup and step<warmup:
            return (step+1)/warmup
        progress=min(1.,max(0.,(step-warmup)/max(1,total-warmup)))
        return .1+.9*.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)


def loss_sum(logits,labels,weights,kind,gamma):
    ce=torch.nn.functional.cross_entropy(logits,labels,reduction='none')
    alpha=weights[labels] if weights is not None else torch.ones_like(ce)
    terms=ce if kind!='focal' else (1-torch.exp(-ce)).pow(gamma)*ce
    return (alpha*terms).sum(),alpha.sum().detach()


def train(config,resume=None):
    config=copy.deepcopy(config)
    t=config['training']; out=Path(config['run']['output_dir'])
    phases=t['phases']
    if not phases or not any(int(p['epochs'])>0 for p in phases) or any(int(p['epochs'])<0 for p in phases):
        raise ValueError('Provide phases with nonnegative epoch counts and at least one epoch')
    if not 0<=t.get('metadata_dropout',0)<=1 or not 0<=t.get('warmup_fraction',.05)<1:
        raise ValueError('Invalid metadata dropout/warmup_fraction')
    if t.get('loss','weighted_ce') not in ('ce','weighted_ce','focal'):
        raise ValueError('loss must be ce | weighted_ce | focal')
    if int(t.get('accumulation_steps',1))<1:
        raise ValueError('accumulation_steps must be positive')
    if (out/'last.pt').exists() and not resume:
        raise FileExistsError('Run already exists; use --resume last.pt or a new output_dir')
    if (out/'best.pt').exists() and not resume:
        raise FileExistsError('Protecting existing best.pt; choose a new output_dir')
    seed_all(t.get('seed',42))
    device=torch.device(t.get('device','cpu'))
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; enable Kaggle GPU or set device: cpu')
    parts,split_info=prepare_splits(config)
    schema=fit_schema(parts['train'],config['data'])
    saved=torch.load(resume,map_location='cpu',weights_only=True) if resume else None
    out.mkdir(parents=True,exist_ok=True)
    if saved:
        old=copy.deepcopy(saved['config']); now=copy.deepcopy(config)
        old['run']['output_dir']=now['run']['output_dir']='IGNORED'
        if old!=now or saved['split_fingerprint']!=split_info['fingerprint'] or saved['schema']!=schema:
            raise ValueError('Resume requires identical configuration/data/splits except output_dir')
        if 'training_state' not in saved:
            raise ValueError('Resume from last.pt, not inference-only best.pt')
        source=Path(resume).parent
        if source.resolve()!=out.resolve():
            if not (source/'best.pt').exists():
                raise FileNotFoundError('Copy best.pt alongside last.pt before resuming')
            if (out/'best.pt').exists() or (out/'last.pt').exists():
                raise FileExistsError('Resume destination must be a fresh run directory')
            shutil.copy2(source/'best.pt',out/'best.pt')
        processor=processor_for(config,source/'processor') if config['model']['backbone']!='tiny' else None
    else:
        processor=processor_for(config)
    if processor:
        processor.save_pretrained(out/'processor')
    for name,frame in parts.items():
        frame.to_csv(out/f'{name}_split.csv',index=False)
    json_save(out/'split_info.json',split_info); json_save(out/'schema.json',schema); json_save(out/'config.json',config)
    model=CrossSkinNet(config['model'],schema,saved['hf_config'] if saved else None).to(device)
    if saved:
        model.load_state_dict(saved['weights'])
    train_loader=loader(parts['train'],schema,processor,config,True)
    val_loader=loader(parts['val'],schema,processor,config)
    counts=parts['train']['_label'].value_counts()
    class_weights=torch.tensor([len(parts['train'])/(len(schema['classes'])*counts[c]) for c in schema['classes']],device=device)
    kind=t.get('loss','weighted_ce'); weights=class_weights if kind in ('weighted_ce','focal') else None
    state=saved['training_state'] if saved else {'next_phase':0,'next_epoch':0,'global_epoch':0,
                                                'best':-1.,'phase_best':-1.,'bad_epochs':0,'history':[]}
    print('Augmentation:',t.get('augmentation',False),'| Loss:',kind,'| Mode:',config['model'].get('mode','gated'))
    if saved:
        restore_rng(saved['rng'])
    accumulation=int(t.get('accumulation_steps',1))
    amp=bool(t.get('amp',True) and device.type=='cuda')
    for phase_index in range(state['next_phase'],len(phases)):
        phase=phases[phase_index]; epochs=int(phase['epochs'])
        if epochs==0:
            continue
        continuation=saved is not None and phase_index==state['next_phase'] and state['next_epoch']>0
        start_epoch=state['next_epoch'] if continuation else 0
        if not continuation:
            state['phase_best']=-1.;state['bad_epochs']=0
        model.configure_phase(int(phase.get('unfreeze_blocks',0)))
        groups=parameter_groups(model,phase)
        optimizer=torch.optim.AdamW(groups,weight_decay=t.get('weight_decay',.01))
        updates=math.ceil(len(train_loader)/accumulation)*epochs
        scheduler=make_scheduler(optimizer,updates,t.get('warmup_fraction',.05))
        scaler=torch.amp.GradScaler('cuda',enabled=amp)
        if continuation:
            optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        print('Phase:',phase['name'],'| unfreeze:',phase.get('unfreeze_blocks',0),'| base LRs:',[g['initial_lr'] for g in optimizer.param_groups])
        for epoch in range(start_epoch,epochs):
            model.train()
            if model.vision is not None:
                model.vision.eval()  # frozen BatchNorm/dropout stable; unfrozen weights still receive gradients
            optimizer.zero_grad(set_to_none=True)
            total_loss=0.; total_den=0.;group_den=0.
            for step,(pixels,values,mask,labels) in enumerate(train_loader):
                mask=mask&(torch.rand(mask.shape)>=t.get('metadata_dropout',0.))
                pixels,values,mask,labels=[v.to(device) for v in (pixels,values,mask,labels)]
                with torch.autocast(device_type=device.type,enabled=amp):
                    logits=model(pixels,values,mask)
                    numerator,denominator=loss_sum(logits,labels,weights,kind,t.get('focal_gamma',2.))
                scaler.scale(numerator).backward()
                den=float(denominator);group_den+=den
                total_loss+=float(numerator.detach());total_den+=den
                if (step+1)%accumulation==0 or step+1==len(train_loader):
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(group_den)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),t.get('gradient_clip',1.))
                    old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()>=old_scale:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True);group_den=0.
            metrics,_,_=evaluate_batches(model,val_loader,device,schema['classes'])
            state['global_epoch']+=1
            record={'epoch':state['global_epoch'],'phase':phase['name'],'phase_epoch':epoch+1,
                    'train_loss':total_loss/total_den,'validation':metrics,
                    'learning_rates':[g['lr'] for g in optimizer.param_groups]}
            state['history'].append(record)
            print(f"Epoch {state['global_epoch']}: loss={record['train_loss']:.4f}, val accuracy={metrics['accuracy']:.4f}, val macro-F1={metrics['macro_f1']:.4f}",flush=True)
            checkpoint={'format_version':1,'weights':model.state_dict(),'config':config,'schema':schema,
                        'hf_config':model.architecture(),'epoch':state['global_epoch'],
                        'validation':metrics,'split_fingerprint':split_info['fingerprint']}
            if metrics['macro_f1']>state['best']:
                state['best']=metrics['macro_f1'];save_atomic(out/'best.pt',checkpoint)
            if metrics['macro_f1']>state['phase_best']+t.get('min_delta',.001):
                state['phase_best']=metrics['macro_f1'];state['bad_epochs']=0
            else:
                state['bad_epochs']+=1
            patience=int(t.get('patience',5))
            stop=patience>0 and state['bad_epochs']>=patience
            done=stop or epoch+1==epochs
            state['next_phase']=phase_index+1 if done else phase_index
            state['next_epoch']=0 if done else epoch+1
            checkpoint.update(training_state=copy.deepcopy(state),optimizer=optimizer.state_dict(),
                              scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state())
            save_atomic(out/'last.pt',checkpoint)
            json_save(out/'history.json',state['history'])
            if stop:
                print('Early stopping this phase; moving to next configured phase.')
                break
        saved=None
    return out/'best.pt'


def load_model(checkpoint,device='cpu'):
    checkpoint=Path(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=CrossSkinNet(saved['config']['model'],saved['schema'],saved['hf_config'])
    model.load_state_dict(saved['weights']);model.to(device).eval()
    processor=processor_for(saved['config'],checkpoint.parent/'processor')
    return model,processor,saved


def confidence_intervals(truth,probs,frame,classes,samples=200,seed=42):
    rng=np.random.default_rng(seed)
    grouped='_group' in frame
    if grouped:
        indices=[np.flatnonzero(frame['_group'].to_numpy()==g) for g in frame['_group'].unique()]
    else:
        indices=[np.array([i]) for i in range(len(frame))]
    values=[]
    for _ in range(samples):
        chosen=np.concatenate([indices[i] for i in rng.integers(0,len(indices),len(indices))])
        y,p=truth[chosen],probs[chosen].argmax(1)
        values.append([accuracy_score(y,p),f1_score(y,p,labels=list(range(len(classes))),average='macro',zero_division=0)])
    a=np.array(values)
    return {'method':'group bootstrap' if grouped else 'image bootstrap',
            'samples':samples,'accuracy_95':np.quantile(a[:,0],[.025,.975]).tolist(),
            'macro_f1_95':np.quantile(a[:,1],[.025,.975]).tolist()}


def evaluate(checkpoint,split='val',device='cpu',stress=False,bootstrap=0):
    if split not in ('val','test'):
        raise ValueError('Only val/test evaluation is supported')
    checkpoint=Path(checkpoint);out=checkpoint.parent
    model,processor,saved=load_model(checkpoint,device)
    config,schema=saved['config'],saved['schema']
    config=copy.deepcopy(config);config['training']['device']=device
    frame=read_csv(out/f'{split}_split.csv')
    batches=loader(frame,schema,processor,config)
    rates=[0.,.25,.5,.75,1.] if stress else [0.]
    results={}
    for rate in rates:
        metrics,truth,probs=evaluate_batches(model,batches,device,schema['classes'],rate,42)
        results[str(rate)]=metrics
        if rate==0:
            predictions=frame[[config['data']['image_column'],'_label']].copy()
            predictions['prediction']=[schema['classes'][i] for i in probs.argmax(1)]
            predictions['confidence']=probs.max(1)
            for i,c in enumerate(schema['classes']):
                predictions[f'p_{c}']=probs[:,i]
            predictions.to_csv(out/f'{split}_predictions.csv',index=False)
            if bootstrap:
                metrics['confidence_intervals']=confidence_intervals(truth,probs,frame,schema['classes'],bootstrap)
        print(f"{split}: metadata drop={rate:.0%}, accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}")
    json_save(out/f'{split}_metrics.json',results)
    plot_reports(out,split,results,schema['classes'])
    return results


def plot_reports(out,split,results,classes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,ax=plt.subplots(figsize=(9,8))
    ConfusionMatrixDisplay(np.array(results['0.0']['confusion_matrix']),display_labels=classes).plot(ax=ax,xticks_rotation=45,colorbar=False)
    fig.tight_layout();fig.savefig(out/f'{split}_confusion.png',dpi=160);plt.close(fig)
    if len(results)>1:
        fig,ax=plt.subplots()
        ax.plot([float(k)*100 for k in results],[v['macro_f1'] for v in results.values()],marker='o')
        ax.set(xlabel='Additional metadata removal probability (%)',ylabel='Macro-F1',title=f'{split}: missing-metadata stress')
        fig.tight_layout();fig.savefig(out/f'{split}_stress.png',dpi=160);plt.close(fig)


def predict(checkpoint,image_path,metadata=None,device='cpu'):
    from PIL import Image
    model,processor,saved=load_model(checkpoint,device)
    with Image.open(image_path) as source:
        image=source.convert('RGB')
        if processor:
            pixels=processor(images=image,return_tensors='pt')['pixel_values']
        else:
            n=saved['config']['model'].get('image_size',32)
            pixels=torch.from_numpy(np.array(image.resize((n,n)),dtype=np.float32)).permute(2,0,1).unsqueeze(0)/255.
    values,mask=encode(metadata or {},saved['schema'])
    with torch.no_grad():
        probs=model(pixels.to(device),values.unsqueeze(0).to(device),mask.unsqueeze(0).to(device)).softmax(1)[0].cpu().tolist()
    classes=saved['schema']['classes'];best=int(np.argmax(probs))
    return {'predicted_class':classes[best],'model_probability':probs[best],
            'probabilities':dict(zip(classes,probs)),'observed_metadata_features':int(mask.sum()),
            'note':'Research model score, not a calibrated clinical diagnosis probability.'}

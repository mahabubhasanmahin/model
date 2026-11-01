"""Controlled validation-only experiments. No test-set selection or auto-relabeling."""
import argparse
import copy
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from crossskinnet.data import digest,json_save,read_csv

ROOT=Path(__file__).resolve().parent


def without_output(config):
    result=copy.deepcopy(config)
    result['run']['output_dir']='<run>'
    return result


def validate_split(config):
    path=Path(config['data']['split_dir'])/'split_info.json'
    if not path.exists():
        raise FileNotFoundError('Run main.py prepare on the base configuration first')
    info=json.loads(path.read_text())
    if info['signature']['data']!=config['data']:
        raise ValueError('Prepared split settings differ from the base configuration')
    for field,hash_key in [('labels_csv','labels_sha256'),('metadata_csv','metadata_sha256')]:
        source=config['data'].get(field)
        if (digest(source) if source else None)!=info['signature'][hash_key]:
            raise ValueError(f'{field} changed after split preparation')
    return info


def make_plan(base_path,directory,baseline_run=None,backbone=None,model_path=None):
    base=yaml.safe_load(Path(base_path).read_text())
    if base['training'].get('augmentation',False) or base['training'].get('loss','weighted_ce')!='weighted_ce':
        raise ValueError('Use the existing augmentation=False, loss=weighted_ce run as base')
    split=validate_split(base)
    destination=Path(directory)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Choose a new empty suite directory; existing plans are protected')
    variants=[('baseline',{},'Existing settings: no augmentation, weighted CE'),
              ('augmentation_on',{'augmentation':True},'Only training augmentation changed'),
              ('ce_loss',{'loss':'ce'},'Only loss changed to plain cross-entropy'),
              ('focal_loss',{'loss':'focal'},'Only loss changed to focal; same default inverse-frequency alpha')]
    cases=[]
    for name,updates,description in variants:
        cfg=copy.deepcopy(base);cfg['training'].update(updates)
        run=destination/'runs'/name
        reuse=False
        if name=='baseline' and baseline_run:
            run=Path(baseline_run)
            original=json.loads((run/'config.json').read_text())
            if without_output(original)!=without_output(cfg):
                raise ValueError('Existing baseline settings differ; use its config.json as base or omit --baseline-run')
            info=json.loads((run/'split_info.json').read_text())
            if info['fingerprint']!=split['fingerprint']:
                raise ValueError('Existing baseline uses different splits')
            if not (run/'best.pt').exists():
                raise FileNotFoundError(run/'best.pt')
            if not completed(run):
                raise ValueError('Existing baseline training is incomplete; finish/resume it before reusing')
            reuse=True
        cfg['run']['output_dir']=str(run)
        cases.append({'name':name,'description':description,'reuse':reuse,'config':cfg})
    if bool(backbone)!=bool(model_path):
        raise ValueError('Supply --backbone and --model-path together')
    if backbone:
        if backbone==base['model']['backbone'] and model_path==base['model']['path']:
            raise ValueError('Alternative model must differ from the baseline')
        cfg=copy.deepcopy(base)
        cfg['model'].update(backbone=backbone,path=model_path,use_fast=backbone=='dinov3')
        name=f'backbone_{backbone}'
        cfg['run']['output_dir']=str(destination/'runs'/name)
        cases.append({'name':name,'description':'Only backbone/checkpoint and its native processor changed',
                      'reuse':False,'config':cfg})
    destination.mkdir(parents=True,exist_ok=True)
    for case in cases:
        path=destination/f"{case['name']}.yaml"
        path.write_text(yaml.safe_dump(case.pop('config'),sort_keys=False))
        case['config_path']=str(path)
    plan={'format_version':1,'split_fingerprint':split['fingerprint'],'cases':cases,
          'selection_metric':'validation macro-F1','note':'Same split/seed; test evaluation is never called by this runner.'}
    json_save(destination/'plan.json',plan)
    print(pd.DataFrame([{k:c[k] for k in ('name','description','reuse')} for c in cases]).to_string(index=False))
    print('Plan:',destination/'plan.json')
    return destination/'plan.json'


def completed(run):
    path=Path(run)/'last.pt'
    if not path.exists():
        return False
    return _completed_cached(str(path.resolve()),path.stat().st_mtime_ns,path.stat().st_size)


@lru_cache(maxsize=128)
def _completed_cached(path,mtime_ns,size):
    state=torch.load(path,map_location='cpu',weights_only=True)
    pointer=state['training_state']['next_phase']
    return all(int(p['epochs'])==0 for p in state['config']['training']['phases'][pointer:])


def run_plan(plan_path,only=None,resume=False):
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text())
    selected=set(only or [c['name'] for c in plan['cases']])
    unknown=selected-{c['name'] for c in plan['cases']}
    if unknown:
        raise ValueError(f'Unknown experiments: {unknown}')
    for case in plan['cases']:
        if case['name'] not in selected:
            continue
        config=yaml.safe_load(Path(case['config_path']).read_text());run=Path(config['run']['output_dir'])
        if validate_split(config)['fingerprint']!=plan['split_fingerprint']:
            raise ValueError('Split fingerprint changed')
        print(f"\n=== {case['name']}: {case['description']} ===",flush=True)
        ready=completed(run)
        if ready:
            original=json.loads((run/'config.json').read_text())
            if original!=config:
                raise ValueError('Finished run settings differ from its planned config')
            print('Reusing finished training; validation will be evaluated again.',flush=True)
        elif case['reuse']:
            raise ValueError('Reusable baseline is no longer a completed run')
        else:
            command=[sys.executable,'-u',str(ROOT/'main.py'),'train','--config',case['config_path']]
            if (run/'last.pt').exists():
                if not resume:
                    raise ValueError('Partial run found. Add --resume to resume it without losing epochs')
                command+=['--resume',str(run/'last.pt')]
            subprocess.run(command,check=True)
        subprocess.run([sys.executable,'-u',str(ROOT/'main.py'),'evaluate','--checkpoint',str(run/'best.pt'),
                        '--split','val','--device',config['training'].get('device','cpu')],check=True)
        summarize(plan_path)
    return summarize(plan_path)


def summarize(plan_path):
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text());rows=[];per_class=[]
    for case in plan['cases']:
        config=yaml.safe_load(Path(case['config_path']).read_text());run=Path(config['run']['output_dir'])
        metric_path=run/'val_metrics.json'
        row={'experiment':case['name'],'run':str(run),'status':'pending'}
        if metric_path.exists() and completed(run):
            saved=json.loads((run/'config.json').read_text())
            if saved!=config:
                raise ValueError(f"Settings changed for {case['name']}")
            info=json.loads((run/'split_info.json').read_text())
            if info['fingerprint']!=plan['split_fingerprint']:
                raise ValueError('Cannot compare different splits')
            metrics=json.loads(metric_path.read_text())['0.0']
            row.update(status='complete',accuracy=metrics['accuracy'],balanced_accuracy=metrics['balanced_accuracy'],macro_f1=metrics['macro_f1'])
            classes=json.loads((run/'schema.json').read_text())['classes']
            for label in classes:
                per_class.append({'experiment':case['name'],'class':label,**metrics['per_class'][label]})
        rows.append(row)
    table=pd.DataFrame(rows)
    if 'macro_f1' in table:
        baseline=table.loc[(table['experiment']=='baseline')&(table['status']=='complete')]
        if len(baseline):
            for metric in ('accuracy','balanced_accuracy','macro_f1'):
                table[f'delta_{metric}']=table[metric]-baseline.iloc[0][metric]
    table.to_csv(plan_path.parent/'validation_comparison.csv',index=False)
    if per_class:
        pd.DataFrame(per_class).to_csv(plan_path.parent/'validation_per_class.csv',index=False)
    print(table.to_string(index=False),flush=True)
    return table


def review_errors(run_dir,max_images=24):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run_dir)
    config=json.loads((run/'config.json').read_text());image_col=config['data']['image_column']
    predictions=read_csv(run/'val_predictions.csv')
    frame=read_csv(run/'val_split.csv')
    errors=predictions[predictions['_label']!=predictions['prediction']].copy()
    errors['confidence']=pd.to_numeric(errors['confidence'])
    errors=errors.merge(frame.drop(columns=['_label']),on=image_col,validate='one_to_one')
    errors=errors.sort_values('confidence',ascending=False)
    errors['reviewed_label']='';errors['image_quality']='';errors['review_notes']=''
    errors.to_csv(run/'validation_errors_review.csv',index=False)
    print('Validation errors:',len(errors),'| CSV:',run/'validation_errors_review.csv')
    if len(errors) and max_images>0:
        sample=errors.head(max_images);cols=4;rows=int(np.ceil(len(sample)/cols))
        fig,axes=plt.subplots(rows,cols,figsize=(12,3.4*rows),squeeze=False)
        for axis in axes.flat:axis.axis('off')
        for axis,(_,row) in zip(axes.flat,sample.iterrows()):
            with Image.open(row['_path']) as image:
                axis.imshow(image.convert('RGB'))
            axis.set_title(f"{row[image_col]}\nTrue: {row['_label']} | Pred: {row['prediction']}\nscore={row['confidence']:.3f}",fontsize=8)
        fig.suptitle('Highest-confidence validation errors — review candidates, not proof of label mistakes')
        fig.tight_layout();fig.savefig(run/'validation_errors_gallery.png',dpi=140);plt.close(fig)
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('--base',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--baseline-run');p.add_argument('--backbone',choices=['dinov3','dinov2','efficientnet','swin'])
    p.add_argument('--model-path')
    p=sub.add_parser('run');p.add_argument('--plan',required=True);p.add_argument('--only',nargs='+');p.add_argument('--resume',action='store_true')
    p=sub.add_parser('summary');p.add_argument('--plan',required=True)
    p=sub.add_parser('review');p.add_argument('--run',required=True);p.add_argument('--max-images',type=int,default=24)
    a=parser.parse_args()
    if a.command=='plan':make_plan(a.base,a.output_dir,a.baseline_run,a.backbone,a.model_path)
    elif a.command=='run':run_plan(a.plan,a.only,a.resume)
    elif a.command=='summary':summarize(a.plan)
    else:review_errors(a.run,a.max_images)


if __name__=='__main__':main()

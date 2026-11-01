import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from crossskinnet.data import prepare_splits
from crossskinnet.model import CrossSkinNet
from crossskinnet.engine import train,evaluate,predict,load_model,parameter_groups


def fixture_config(tmp_path):
    torch.set_num_threads(1)
    rows=[]
    for i in range(120):
        name=f'img{i:03}'
        Image.new('RGB',(24,24),(180*(i%2),30,50)).save(tmp_path/f'{name}.jpg')
        rows.append({'image':name,'diagnosis':str(i%2),'lesion_id':f'L{i//2:03}',
                     'age':30+i if i%3 else np.nan,'sex':'F' if i%2 else None})
    pd.DataFrame(rows).to_csv(tmp_path/'labels.csv',index=False)
    return {'model':{'backbone':'tiny','path':'unused','mode':'gated','image_size':16},
            'data':{'image_root':str(tmp_path),'labels_csv':str(tmp_path/'labels.csv'),
                    'metadata_csv':None,'image_column':'image','label_column':'diagnosis','label_columns':None,
                    'numeric_features':['age'],'categorical_features':['sex'],'group_columns':['lesion_id'],
                    'missing_group_policy':'error','split_dir':str(tmp_path/'splits'),'verify_images':True},
            'training':{'device':'cpu','batch_size':8,'num_workers':0,'seed':42,'augmentation':True,
                        'loss':'weighted_ce','metadata_dropout':.1,'accumulation_steps':2,'patience':0,
                        'phases':[{'name':'head','epochs':1,'unfreeze_blocks':0,'head_lr':.001,'backbone_lr':.00001},
                                  {'name':'fine','epochs':2,'unfreeze_blocks':2,'head_lr':.0001,'backbone_lr':.00001}]},
            'run':{'output_dir':str(tmp_path/'run')}}


def test_train_resume_prediction_and_stress(tmp_path,monkeypatch):
    import crossskinnet.engine as engine
    cfg=fixture_config(tmp_path)
    full=train(cfg)
    full_last=torch.load(full.parent/'last.pt',weights_only=True)
    partial=copy.deepcopy(cfg);partial['run']['output_dir']=str(tmp_path/'partial')
    actual_save=engine.save_atomic
    class Interrupt(Exception): pass
    def interrupted_save(path,state):
        actual_save(path,state)
        if Path(path).name=='last.pt' and state['training_state']['global_epoch']==2:
            raise Interrupt()
    monkeypatch.setattr(engine,'save_atomic',interrupted_save)
    with pytest.raises(Interrupt):
        train(partial)
    monkeypatch.setattr(engine,'save_atomic',actual_save)
    resumed=train(partial,Path(partial['run']['output_dir'])/'last.pt')
    resumed_last=torch.load(resumed.parent/'last.pt',weights_only=True)
    for name,weight in full_last['weights'].items():
        torch.testing.assert_close(weight,resumed_last['weights'][name],rtol=0,atol=0)
    assert full_last['training_state']['history']==resumed_last['training_state']['history']
    result=evaluate(resumed,'val',stress=True,bootstrap=5)
    assert len(result)==5
    assert (resumed.parent/'val_confusion.png').exists()
    prediction=predict(resumed,tmp_path/'img000.jpg',{'age':40})
    assert abs(sum(prediction['probabilities'].values())-1)<1e-6
    assert prediction['observed_metadata_features']==1
    model,_,saved=load_model(resumed)
    pixels=torch.randn(2,3,16,16);values=torch.full((2,2),float('nan'));mask=torch.zeros(2,2,dtype=torch.bool)
    torch.testing.assert_close(model(pixels,values,mask),model.head(model.image_features(pixels)))
    with pytest.raises(FileExistsError):
        train(partial)
    changed=copy.deepcopy(partial);changed['training']['augmentation']=False
    with pytest.raises(ValueError,match='identical'):
        train(changed,resumed.parent/'last.pt')


def test_group_filtering_and_split_identity(tmp_path):
    cfg=fixture_config(tmp_path)
    path=tmp_path/'labels.csv';frame=pd.read_csv(path,dtype=str)
    frame.loc[0,'lesion_id']=None;frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='missing group'):
        prepare_splits(cfg)
    cfg['data']['missing_group_policy']='drop'
    parts,info=prepare_splits(cfg)
    assert info['audit']['excluded_missing_group']==1
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not set(parts[a]['lesion_id'])&set(parts[b]['lesion_id'])
    again,_=prepare_splits(cfg)
    assert parts['train']['image'].tolist()==again['train']['image'].tolist()
    changed=copy.deepcopy(cfg);changed['data']['split_seed']=99
    with pytest.raises(ValueError,match='different data'):
        prepare_splits(changed)


@pytest.mark.parametrize('mode',['image_only','metadata_only','naive'])
def test_ablation_modes(tmp_path,mode):
    cfg=fixture_config(tmp_path);cfg['model']['mode']=mode
    cfg['training']['phases']=cfg['training']['phases'][:1]
    best=train(cfg)
    assert 0<=evaluate(best,'val')['0.0']['accuracy']<=1


@pytest.mark.parametrize('kind',['dinov3','dinov2','swin','efficientnet'])
def test_real_backbone_and_gradcam(kind):
    from transformers import DINOv3ViTConfig,Dinov2Config,SwinConfig,EfficientNetConfig
    from crossskinnet.xai import gradcam
    torch.set_num_threads(1)
    if kind=='dinov3':
        h=DINOv3ViTConfig(image_size=32,patch_size=8,hidden_size=32,num_hidden_layers=4,num_attention_heads=4)
    elif kind=='dinov2':
        h=Dinov2Config(image_size=32,patch_size=8,hidden_size=32,num_hidden_layers=4,num_attention_heads=4)
    elif kind=='swin':
        h=SwinConfig(image_size=32,embed_dim=8,depths=[2,2],num_heads=[2,4],window_size=4)
    else:
        h=EfficientNetConfig(image_size=32,width_coefficient=1.,depth_coefficient=1.,hidden_dim=1280)
    schema={'classes':['a','b'],'numeric':[],'categorical':[]}
    model=CrossSkinNet({'backbone':kind,'mode':'image_only'},schema,h.to_dict()).eval()
    model.configure_phase(2)
    groups=parameter_groups(model,{'head_lr':1e-4,'backbone_lr':1e-5})
    assert [g['lr'] for g in groups]==[1e-4,1e-5]
    pixels=torch.randn(1,3,32,32);values=torch.empty(1,0);mask=torch.empty(1,0,dtype=torch.bool)
    assert model(pixels,values,mask).shape==(1,2)
    model.configure_phase(4)
    meaningful=False
    for label in (0,1):
        try:
            heat,target=gradcam(model,pixels,values,mask,label)
            assert heat.ndim==2 and np.isfinite(heat).all()
            meaningful=True;break
        except RuntimeError as exc:
            if 'Degenerate/zero' not in str(exc): raise
    # A 32px EfficientNet has a 1x1 final grid; an informative spatial CAM
    # is not guaranteed for an untrained network. Degeneracy must be explicit.
    if kind != 'efficientnet':
        assert meaningful


def test_local_dinov3_fast_processor_roundtrip(tmp_path):
    from transformers import DINOv3ViTConfig,DINOv3ViTModel,DINOv3ViTImageProcessorFast
    cfg=fixture_config(tmp_path)
    path=tmp_path/'local_dinov3'
    h=DINOv3ViTConfig(image_size=32,patch_size=8,hidden_size=32,num_hidden_layers=2,num_attention_heads=4)
    DINOv3ViTModel(h).save_pretrained(path)
    DINOv3ViTImageProcessorFast(size={'height':32,'width':32}).save_pretrained(path)
    cfg['model'].update(backbone='dinov3',path=str(path),use_fast=True)
    cfg['training']['phases'][1]['epochs']=1
    best=train(cfg)
    assert predict(best,tmp_path/'img000.jpg')['observed_metadata_features']==0
    assert 0<=evaluate(best,'val')['0.0']['macro_f1']<=1


def test_optional_shap_and_cam(tmp_path):
    pytest.importorskip('shap')
    from crossskinnet.xai import explain
    cfg=fixture_config(tmp_path)
    cfg['training']['phases']=cfg['training']['phases'][:1]
    best=train(cfg)
    success=False
    for label in (0,1):
        try:
            out=tmp_path/f'explain{label}'
            explain(best,tmp_path/'img001.jpg',{'age':45,'sex':'F'},'cpu',out,label,True)
            values=pd.read_csv(out/'metadata_shap.csv')
            assert len(values)==2 and np.isfinite(values['shap_value']).all()
            assert (out/'gradcam.png').exists()
            success=True;break
        except RuntimeError as exc:
            if 'Degenerate/zero' not in str(exc): raise
    assert success

import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def json_save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2),encoding='utf8')


def read_csv(path):
    # Preserve IDs/categories such as 001. Numeric conversion happens explicitly.
    return pd.read_csv(path,dtype=str)


def table_from_config(cfg):
    print('[prepare] Reading labels and metadata...', flush=True)
    frame=read_csv(cfg['labels_csv'])
    image=cfg['image_column']
    if frame[image].isna().any() or frame[image].duplicated().any():
        raise ValueError('Image IDs must be unique and nonmissing')
    if cfg.get('label_columns'):
        hot=frame[cfg['label_columns']].apply(pd.to_numeric,errors='raise')
        if not (hot.isin([0,1]).all(axis=1)&hot.sum(axis=1).eq(1)).all():
            raise ValueError('One-hot rows must contain exactly one positive; check unknown-class rows')
        frame['_label']=hot.idxmax(axis=1)
    else:
        frame['_label']=frame[cfg['label_column']]
        if frame['_label'].isna().any():
            raise ValueError('Missing disease labels')
    if cfg.get('metadata_csv'):
        meta=read_csv(cfg['metadata_csv'])
        overlap=(set(frame)&set(meta))-{image}
        if overlap:
            raise ValueError(f'Duplicate metadata columns: {overlap}')
        frame=frame.merge(meta,on=image,how='left',validate='one_to_one')
    for name in cfg.get('numeric_features',[])+cfg.get('categorical_features',[]):
        if name not in frame:
            print(f'WARNING: absent feature {name}; treated as missing')
            frame[name]=np.nan
    for name in cfg.get('numeric_features',[]):
        frame[name]=pd.to_numeric(frame[name],errors='raise')
        if np.isinf(frame[name].dropna()).any():
            raise ValueError(f'Infinite numeric values: {name}')
    index={}
    root=Path(cfg['image_root'])
    if not root.is_dir():
        raise FileNotFoundError(root)
    for p in tqdm(root.rglob('*'),desc='Scanning image files',unit='file',mininterval=1):
        if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp'):
            if p.stem in index:
                raise ValueError(f'Duplicate image basename {p.stem}; use a narrower image_root')
            index[p.stem]=str(p.resolve())
    def resolve(value):
        p=root/value
        if p.is_file():
            return str(p.resolve())
        if Path(value).stem not in index:
            raise FileNotFoundError(value)
        return index[Path(value).stem]
    frame['_path']=[resolve(value) for value in tqdm(frame[image],total=len(frame),
                                                    desc='Resolving image paths',unit='image',mininterval=1)]
    if frame['_path'].duplicated().any():
        raise ValueError('Duplicate resolved image paths')
    report={'original_images':len(frame),'excluded_missing_group':0,
            'original_class_counts':frame['_label'].value_counts().to_dict()}
    groups=cfg.get('group_columns',[])
    for name in groups:
        if name not in frame:
            raise ValueError(f'Group identifier {name} absent; do not substitute image IDs for patient IDs')
    if groups:
        missing=frame[groups].isna().any(axis=1)|frame[groups].apply(lambda c:c.str.strip().eq('')).any(axis=1)
        report['excluded_missing_group']=int(missing.sum())
        report['excluded_class_counts']=frame.loc[missing,'_label'].value_counts().to_dict()
        if missing.any():
            if cfg.get('missing_group_policy','error')!='drop':
                raise ValueError(f'{missing.sum()} rows have missing group IDs; choose explicit drop policy or supply mappings')
            frame=frame.loc[~missing].copy()
        if frame.empty:
            raise ValueError('No images remain after group filtering')
        # Connected components preserve BOTH patient and lesion links if supplied.
        frame=frame.reset_index(drop=True)
        parent=list(range(len(frame)))
        def find(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]]
                i=parent[i]
            return i
        for name in groups:
            seen={}
            for i,v in tqdm(enumerate(frame[name]),total=len(frame),
                            desc=f'Grouping by {name}',unit='image',mininterval=1):
                if v in seen:
                    parent[find(i)]=find(seen[v])
                else:
                    seen[v]=i
        frame['_group']=[str(find(i)) for i in range(len(frame))]
        report['group_columns']=groups
        report['unique_connected_groups']=frame['_group'].nunique()
        report['independence']='grouped by supplied identifiers; unknown patient relationships remain unverified'
    elif not cfg.get('allow_image_split',False):
        raise ValueError('Provide group_columns, or explicitly allow an exploratory image-level split')
    else:
        report['independence']='IMAGE LEVEL ONLY: patient/lesion independence unverified'
        print('WARNING:',report['independence'])
    if cfg.get('verify_images',True):
        for path in tqdm(frame['_path'],total=len(frame),desc='Verifying images',unit='image',mininterval=1):
            with Image.open(path) as image_file:
                image_file.verify()
    if cfg.get('check_exact_duplicates',False):
        # Link exact duplicate content even when filenames differ.
        hashes=pd.Series([digest(path) for path in tqdm(frame['_path'],total=len(frame),
                                                        desc='Checking exact duplicates',unit='image',mininterval=1)],index=frame.index)
        if hashes.duplicated().any():
            raise ValueError('Exact duplicate image content detected; resolve duplicates before splitting')
        report['exact_duplicate_check']='passed'
    else:
        report['exact_duplicate_check']='not run; near-duplicate detection also not provided'
    report['included_images']=len(frame)
    report['included_class_counts']=frame['_label'].value_counts().to_dict()
    return frame.reset_index(drop=True),report


def assert_splits(parts,cfg):
    names=['train','val','test']
    classes=set(parts['train']['_label'])
    if len(classes)<2:
        raise ValueError('Need at least two training classes')
    for name in names:
        if not len(parts[name]) or set(parts[name]['_label'])!=classes:
            raise ValueError(f'{name} must contain all training classes; revise split strategy, not based on model scores')
    for i,a in enumerate(names):
        for b in names[i+1:]:
            for column in ['_path']+cfg.get('group_columns',[]):
                if set(parts[a][column])&set(parts[b][column]):
                    raise ValueError(f'{column} overlap in {a}/{b}')


def prepare_splits(config):
    cfg=config['data']
    directory=Path(cfg['split_dir'])
    print('[prepare] Checking CSV fingerprints and saved splits...', flush=True)
    signature={'data':cfg,'labels_sha256':digest(cfg['labels_csv']),
               'metadata_sha256':digest(cfg['metadata_csv']) if cfg.get('metadata_csv') else None}
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    marker=directory/'split_info.json'
    if marker.exists():
        info=json.loads(marker.read_text())
        if info['fingerprint']!=fingerprint:
            raise ValueError('split_dir belongs to different data/settings; do not overwrite it')
        parts={s:read_csv(directory/f'{s}.csv') for s in ['train','val','test']}
        assert_splits(parts,cfg)
        print('Reusing fixed splits:',directory,flush=True)
        return parts,info
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Incomplete/nonempty split_dir; inspect it before creating new splits')
    frame,report=table_from_config(cfg)
    seed=cfg.get('split_seed',42)
    split_column=cfg.get('split_column')
    if split_column:
        if not frame[split_column].isin(['train','val','test']).all():
            raise ValueError('split_column must use train/val/test')
        parts={s:frame[frame[split_column]==s].copy() for s in ['train','val','test']}
    elif cfg.get('group_columns'):
        # Label-stratified group folds: approximately 70/10/20 by image count.
        print('[prepare] Creating grouped test split (5 folds)...',flush=True)
        outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
        a,b=next(outer.split(frame,frame['_label'],frame['_group']))
        rest,test=frame.iloc[a],frame.iloc[b]
        print('[prepare] Creating grouped validation split (8 folds)...',flush=True)
        inner=StratifiedGroupKFold(n_splits=8,shuffle=True,random_state=seed)
        a,b=next(inner.split(rest,rest['_label'],rest['_group']))
        parts={'train':rest.iloc[a].copy(),'val':rest.iloc[b].copy(),'test':test.copy()}
    else:
        rest,test=train_test_split(frame,test_size=.2,stratify=frame['_label'],random_state=seed)
        train,val=train_test_split(rest,test_size=.125,stratify=rest['_label'],random_state=seed)
        parts={'train':train,'val':val,'test':test}
    print('[prepare] Checking split independence and saving manifests...',flush=True)
    assert_splits(parts,cfg)
    directory.mkdir(parents=True,exist_ok=True)
    info={'fingerprint':fingerprint,'signature':signature,'audit':report,'splits':{}}
    for name,table in parts.items():
        table.to_csv(directory/f'{name}.csv',index=False)
        info['splits'][name]={'images':len(table),'class_counts':table['_label'].value_counts().to_dict()}
    json_save(marker,info)
    print(json.dumps(info['audit'],indent=2),flush=True)
    print('[prepare] Complete. Fixed split files saved to:',directory,flush=True)
    return parts,info


def fit_schema(frame,cfg):
    result={'classes':sorted(frame['_label'].unique().tolist()),'numeric':[],'categorical':[]}
    for name in cfg.get('numeric_features',[]):
        v=pd.to_numeric(frame[name],errors='raise').dropna()
        result['numeric'].append({'name':name,'mean':float(v.mean()) if len(v) else 0.,
                                  'std':max(float(v.std(ddof=0)),1e-8) if len(v) else 1.})
    for name in cfg.get('categorical_features',[]):
        result['categorical'].append({'name':name,'vocab':sorted(frame[name].dropna().astype(str).unique().tolist())})
    return result


def encode(row,schema):
    values,mask=[],[]
    for f in schema['numeric']:
        value=row.get(f['name'])
        ok=pd.notna(value) and str(value).strip()!=''
        number=float(value) if ok else f['mean']
        if not np.isfinite(number):
            raise ValueError(f'Nonfinite metadata: {f["name"]}')
        values.append((number-f['mean'])/f['std'] if ok else 0.)
        mask.append(ok)
    for f in schema['categorical']:
        value=row.get(f['name'])
        ok=pd.notna(value) and str(value).strip()!=''
        values.append(f['vocab'].index(str(value))+1 if ok and str(value) in f['vocab'] else 0)
        mask.append(ok)
    return torch.tensor(values,dtype=torch.float32),torch.tensor(mask,dtype=torch.bool)


class SkinDataset(Dataset):
    def __init__(self,frame,schema,processor,augmentation=False,image_size=32):
        self.rows=frame.to_dict('records')
        self.schema,self.processor,self.augmentation,self.image_size=schema,processor,augmentation,image_size
        self.labels={c:i for i,c in enumerate(schema['classes'])}
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row['_path']) as source:
            image=source.convert('RGB')
            if self.augmentation:
                if random.random()<.5:
                    image=ImageOps.mirror(image)
                if random.random()<.5:
                    image=ImageOps.flip(image)
                image=image.rotate(random.choice([0,90,180,270]),expand=True)
            if self.processor is None:
                pixels=torch.from_numpy(np.array(image.resize((self.image_size,self.image_size)),dtype=np.float32)).permute(2,0,1)/255.
            else:
                pixels=self.processor(images=image,return_tensors='pt')['pixel_values'][0]
        values,mask=encode(row,self.schema)
        return pixels,values,mask,self.labels[row['_label']]


def seed_worker(_):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed)
    random.seed(seed)


def loader(frame,schema,processor,config,training=False):
    t=config['training']
    return DataLoader(SkinDataset(frame,schema,processor,training and t.get('augmentation',False),
                                  config['model'].get('image_size',32)),
                      batch_size=t.get('batch_size',8),shuffle=training,
                      num_workers=t.get('num_workers',0),pin_memory=t.get('device','cpu').startswith('cuda'),
                      worker_init_fn=seed_worker)

"""Offline end-to-end check with synthetic data, not a medical accuracy test."""
import tempfile
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from crossskinnet.engine import train,evaluate,predict


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);rows=[]
        for i in range(120):
            name=f'image{i:03}'
            Image.new('RGB',(24,24),(200*(i%2),40,60)).save(root/f'{name}.jpg')
            rows.append({'image':name,'label':str(i%2),'lesion_id':f'lesion{i//2:03}',
                         'age':30+i if i%3 else None,'sex':'F' if i%2 else None})
        pd.DataFrame(rows).to_csv(root/'labels.csv',index=False)
        config={'model':{'backbone':'tiny','path':'unused','mode':'gated','image_size':16},
                'data':{'labels_csv':str(root/'labels.csv'),'metadata_csv':None,'image_root':str(root),
                        'image_column':'image','label_column':'label','numeric_features':['age'],
                        'categorical_features':['sex'],'group_columns':['lesion_id'],
                        'split_dir':str(root/'splits'),'verify_images':True},
                'training':{'device':'cpu','batch_size':8,'num_workers':0,'augmentation':True,
                            'accumulation_steps':2,'loss':'weighted_ce','metadata_dropout':.1,
                            'phases':[{'name':'head','epochs':1,'unfreeze_blocks':0,'head_lr':.001,'backbone_lr':1e-5},
                                      {'name':'fine','epochs':1,'unfreeze_blocks':2,'head_lr':1e-4,'backbone_lr':1e-5}]},
                'run':{'output_dir':str(root/'run')}}
        best=train(config)
        results=evaluate(best,'val',stress=True)
        assert len(results)==5
        output=predict(best,root/'image000.jpg',{'age':45})
        assert output['observed_metadata_features']==1
    print('PASS: synthetic train, checkpoint reload, evaluation, stress test, single-image prediction.')


if __name__=='__main__':
    main()

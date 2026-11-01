import argparse
import json
from pathlib import Path
from crossskinnet.config import read_config
from crossskinnet.data import prepare_splits
from crossskinnet.engine import train,evaluate,predict


def main():
    parser=argparse.ArgumentParser(description='CrossSkinNet research pipeline')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','train'):
        p=sub.add_parser(name);p.add_argument('--config',required=True)
        if name=='train':
            p.add_argument('--resume',help='last.pt; identical settings except output_dir required')
    p=sub.add_parser('evaluate');p.add_argument('--checkpoint',required=True)
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--device',default='cpu')
    p.add_argument('--stress',action='store_true');p.add_argument('--bootstrap',type=int,default=0)
    p=sub.add_parser('predict');p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True)
    p.add_argument('--metadata',help='JSON file with raw metadata values; omitted features are missing')
    p.add_argument('--device',default='cpu');p.add_argument('--output')
    p=sub.add_parser('xai');p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True)
    p.add_argument('--metadata');p.add_argument('--device',default='cpu');p.add_argument('--class-index',type=int)
    p.add_argument('--output-dir',required=True);p.add_argument('--shap',action='store_true')
    p=sub.add_parser('compare');p.add_argument('runs',nargs='+',help='run output folders; compares validation only')
    args=parser.parse_args()
    if args.command in ('prepare','train'):
        config=read_config(args.config)
        if args.command=='prepare':
            parts,info=prepare_splits(config)
            print(json.dumps(info,indent=2))
        else:
            print('Best checkpoint:',train(config,args.resume))
    elif args.command=='evaluate':
        evaluate(args.checkpoint,args.split,args.device,args.stress,args.bootstrap)
    elif args.command=='predict':
        metadata=json.loads(Path(args.metadata).read_text()) if args.metadata else {}
        result=predict(args.checkpoint,args.image,metadata,args.device)
        print(json.dumps(result,indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(result,indent=2))
    elif args.command=='xai':
        from crossskinnet.xai import explain
        metadata=json.loads(Path(args.metadata).read_text()) if args.metadata else {}
        explain(args.checkpoint,args.image,metadata,args.device,args.output_dir,args.class_index,args.shap)
    elif args.command=='compare':
        import pandas as pd
        rows=[];fingerprints=set()
        for directory in args.runs:
            root=Path(directory)
            info=json.loads((root/'split_info.json').read_text());fingerprints.add(info['fingerprint'])
            metrics=root/'val_metrics.json'
            if not metrics.exists():
                raise FileNotFoundError(f'Run validation evaluation first: {metrics}')
            m=json.loads(metrics.read_text())['0.0']
            rows.append({'run':str(root),'accuracy':m['accuracy'],'balanced_accuracy':m['balanced_accuracy'],'macro_f1':m['macro_f1']})
        if len(fingerprints)!=1:
            raise ValueError('Runs use different splits; direct comparison is invalid')
        print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    main()

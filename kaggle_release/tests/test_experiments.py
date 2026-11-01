import copy
import json
from pathlib import Path
import pandas as pd
import pytest
import yaml
from test_release import fixture_config
from crossskinnet.engine import train,evaluate
from experiments import make_plan,run_plan,summarize,review_errors


def test_controlled_suite_and_error_review(tmp_path):
    cfg=fixture_config(tmp_path)
    cfg['training']['augmentation']=False
    cfg['training']['phases']=cfg['training']['phases'][:1]
    baseline=train(cfg)
    evaluate(baseline,'val')
    base=tmp_path/'base.yaml';base.write_text(yaml.safe_dump(cfg))
    plan_path=make_plan(base,tmp_path/'suite',baseline.parent,'dinov2','not-downloaded-in-this-test')
    plan=json.loads(plan_path.read_text())
    assert {c['name'] for c in plan['cases']}=={'baseline','augmentation_on','ce_loss','focal_loss','backbone_dinov2'}
    # Every comparison keeps exactly the same split/data settings and seed.
    for case in plan['cases']:
        candidate=yaml.safe_load(Path(case['config_path']).read_text())
        assert candidate['data']==cfg['data']
        assert candidate['training']['seed']==cfg['training']['seed']
    results=run_plan(plan_path,only=['baseline','augmentation_on'])
    assert (results['status']=='complete').sum()==2
    assert (tmp_path/'suite'/'validation_comparison.csv').exists()
    assert not (baseline.parent/'test_metrics.json').exists()
    assert not (tmp_path/'suite'/'runs'/'augmentation_on'/'test_metrics.json').exists()
    # Running a completed case again reuses training, with no extra epochs.
    history=(tmp_path/'suite'/'runs'/'augmentation_on'/'history.json').read_text()
    run_plan(plan_path,only=['augmentation_on'])
    assert (tmp_path/'suite'/'runs'/'augmentation_on'/'history.json').read_text()==history
    # Inject a deliberate prediction error ONLY in this test fixture to exercise gallery export.
    p=baseline.parent/'val_predictions.csv';preds=pd.read_csv(p,dtype=str)
    preds.loc[0,'prediction']='1' if preds.loc[0,'_label']=='0' else '0'
    preds.to_csv(p,index=False)
    errors=review_errors(baseline.parent,max_images=2)
    assert len(errors)>0 and (baseline.parent/'validation_errors_gallery.png').exists()
    assert 'review_notes' in errors
    with pytest.raises(ValueError,match='Unknown'):
        run_plan(plan_path,only=['test_set'])
    # A different split's metrics are rejected rather than compared.
    info_path=baseline.parent/'split_info.json';info=json.loads(info_path.read_text());info['fingerprint']='wrong'
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError,match='different splits'):
        summarize(plan_path)

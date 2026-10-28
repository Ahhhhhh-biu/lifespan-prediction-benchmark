"""Repeatable software checks. No real lifespan dataset is used."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark import run

ROOT=Path(__file__).resolve().parent
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);rng=np.random.default_rng(42);n=72
        x=rng.normal(size=(n,3));y=20+2*x[:,0]-x[:,1]+rng.normal(size=n)
        df=pd.DataFrame({'sample_id':range(n),'condition_id':np.repeat(np.arange(12),6),
            'lifespan_days':y,'pharynx_pathology':x[:,0],'intestinal_pathology':x[:,1],'measurement_age_days':x[:,2]})
        data=tmp/'synthetic.csv';df.to_csv(data,index=False)
        cfg=json.loads((ROOT/'benchmark.example.json').read_text());cfg['synthetic_only']=True
        config=tmp/'config.json';config.write_text(json.dumps(cfg))
        for name in ['one','two']:run(data,config,tmp/name)
        for filename in ['oof_predictions.csv','fold_metrics.csv','heldout_permutation_importance.csv','split_manifest.json']:
            assert digest(tmp/'one'/filename)==digest(tmp/'two'/filename),filename
        manifest=json.loads((tmp/'one/split_manifest.json').read_text())
        assert all(not set(x['train_groups']) & set(x['test_groups']) for x in manifest)
        pred=pd.read_csv(tmp/'one/oof_predictions.csv')
        assert len(pred)==n*3 and pred.groupby(['model','row_id']).size().eq(1).all()
        cfg['features'].append('lifespan_days');config.write_text(json.dumps(cfg))
        try:run(data,config,tmp/'bad')
        except ValueError:pass
        else:raise AssertionError('Target included as feature was not rejected')
        report={'status':'passed','scope':'synthetic regression software validation',
            'checks':['group-disjoint outer folds','one OOF prediction per row per model',
            'same-seed benchmark output reproducibility','explicit target-feature rejection'],
        }
        (ROOT/'software_checks.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
if __name__=='__main__':main()

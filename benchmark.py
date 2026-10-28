"""Grouped nested-CV regression scaffold. Synthetic tests validate software behavior."""
import argparse, hashlib, json, platform
from pathlib import Path
import importlib.metadata as md
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run(csv_path, config_path, out_path):
    cfg = json.loads(Path(config_path).read_text())
    out = Path(out_path); out.mkdir(parents=True, exist_ok=False)
    df = pd.read_csv(csv_path)
    features = cfg['features']; target = cfg['target']; group = cfg['group']; rowid = cfg['row_id']
    if len(set([target, group, rowid] + features)) != len(features) + 3:
        raise ValueError('Features must be unique and exclude target, group and row ID')
    if df[rowid].isna().any() or df[rowid].duplicated().any():
        raise ValueError('Row IDs must be unique and non-null')
    if df[group].isna().any() or df[target].isna().any():
        raise ValueError('Missing groups or targets require a documented upstream decision')
    X = df[features].apply(pd.to_numeric, errors='raise')
    y = pd.to_numeric(df[target], errors='raise').to_numpy()
    groups = df[group].astype(str).to_numpy()
    if np.isinf(X.to_numpy()).any() or not np.isfinite(y).all():
        raise ValueError('Infinite features or nonfinite targets')
    if any(X[col].isna().all() for col in features):
        raise ValueError('All-missing feature')
    if len(np.unique(groups)) < 6:
        raise ValueError('At least six groups needed for this demonstration protocol')
    seed = cfg.get('seed', 42)
    outer = GroupKFold(n_splits=cfg.get('outer_folds', 5))
    models = {
        'dummy': (DummyRegressor(), {}),
        'ridge': (Ridge(), {'model__alpha': [0.1, 1.0, 10.0, 100.0]}),
        'random_forest': (RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
                          {'model__max_depth': [3, None], 'model__min_samples_leaf': [2, 5]})
    }
    preds=[]; metrics=[]; manifests=[]; importances=[]
    for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
        assert not set(groups[tr]) & set(groups[te])
        if X.iloc[tr].isna().all().any():
            raise ValueError('An outer training fold has an all-missing feature')
        manifests.append({'fold': fold, 'train_groups': sorted(set(groups[tr])),
                          'test_groups': sorted(set(groups[te])),
                          'train_row_ids': df[rowid].iloc[tr].tolist(),
                          'test_row_ids': df[rowid].iloc[te].tolist()})
        for name, (est, grid) in models.items():
            pipe = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                             ('scale', StandardScaler()), ('model', est)])
            search = GridSearchCV(pipe, grid, scoring='neg_mean_absolute_error',
                                  cv=GroupKFold(n_splits=cfg.get('inner_folds', 3)), n_jobs=1)
            search.fit(X.iloc[tr], y[tr], groups=groups[tr])
            pred = search.predict(X.iloc[te])
            metrics.append({'fold': fold, 'model': name, 'n_test': len(te),
                            'mae': mean_absolute_error(y[te], pred),
                            'r2': r2_score(y[te], pred) if len(te)>1 else None,
                            'best_params': json.dumps(search.best_params_)})
            for i, value in zip(te, pred):
                preds.append({'row_id': df[rowid].iloc[i], 'group': groups[i], 'fold': fold,
                              'model': name, 'observed': float(y[i]), 'predicted': float(value)})
            if name != 'dummy':
                imp = permutation_importance(search.best_estimator_, X.iloc[te], y[te],
                       scoring='neg_mean_absolute_error', n_repeats=5, random_state=seed)
                for feat, value in zip(features, imp.importances_mean):
                    importances.append({'fold': fold, 'model': name, 'feature': feat,
                                        'mean_mae_increase': float(value)})
    predictions = pd.DataFrame(preds)
    predictions.to_csv(out/'oof_predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(out/'fold_metrics.csv', index=False)
    pd.DataFrame(importances).to_csv(out/'heldout_permutation_importance.csv', index=False)
    summaries=[]
    for name, rows in predictions.groupby('model'):
        group_errors = rows.assign(ae=abs(rows.observed-rows.predicted)).groupby('group').ae.mean().to_numpy()
        rng=np.random.default_rng(seed)
        boot=np.array([rng.choice(group_errors, len(group_errors), replace=True).mean() for _ in range(1000)])
        summaries.append({'model':name, 'row_weighted_mae':mean_absolute_error(rows.observed,rows.predicted),
                          'group_equal_weight_mae':float(group_errors.mean()),
                          'group_bootstrap_mae_95pct_ci':np.quantile(boot,[.025,.975]).tolist(),
                          'oof_r2':r2_score(rows.observed,rows.predicted)})
    meta={'status':'synthetic_smoke_test' if cfg.get('synthetic_only') else 'requires_biological_review',
          'input_sha256':hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
          'config':cfg, 'python':platform.python_version(),
          'versions':{k:md.version(k) for k in ['numpy','pandas','scikit-learn']},
          'summary':summaries,
          'limitations':['OOF estimates are not an independent external test.',
           'Intervals describe sampled groups and do not include all model-selection uncertainty.',
           'Held-out importance is exploratory; correlated features and within-group rows limit inference.',
           'User must audit group definitions, feature timing, target leakage and data rights.']}
    (out/'results.json').write_text(json.dumps(meta,indent=2))
    (out/'split_manifest.json').write_text(json.dumps(manifests,indent=2))
    print(json.dumps({'models':len(summaries),'outer_folds':len(manifests),'status':meta['status']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.data,a.config,a.out)

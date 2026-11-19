"""Leakage-controlled forecasting experiment on the Brazil state-week dengue panel.
Direct h-step forecasts (h=1,2,4). Models predict y_{t+h}-y_t (log1p scale); features use data up to origin t only."""
import pandas as pd, numpy as np, warnings, sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')
rng=np.random.default_rng(42)
P=pd.read_csv('panel_state_week.csv',parse_dates=['wk']).sort_values(['state','wk']).reset_index(drop=True)
P['y']=np.log1p(P.cases)
H=[1,2,4]
FEATS=['l0','l1','l2','l3','l4','l5','l6','l7','m4','m8','d1','d4','seas','sin','cos']
def make(h):
    out=[]
    for s,g in P.groupby('state'):
        g=g.copy(); y=g.y
        for k in range(8): g[f'l{k}']=y.shift(k)
        g['m4']=y.rolling(4).mean(); g['m8']=y.rolling(8).mean()
        g['d1']=y-y.shift(1); g['d4']=y-y.shift(4)
        g['seas']=y.shift(52-h)                 # value observed 52 weeks before the target week; known at origin
        woy=g.wk.dt.isocalendar().week.astype(int)
        g['sin']=np.sin(2*np.pi*woy/52.18); g['cos']=np.cos(2*np.pi*woy/52.18)
        g['target']=y.shift(-h); g['tdate']=g.wk.shift(-h)
        out.append(g)
    D=pd.concat(out).dropna(subset=FEATS+['target']).reset_index(drop=True)
    D['delta']=D.target-D.l0
    return D
def models():
    return {'Ridge-AR':lambda: make_pipeline(StandardScaler(),Ridge(alpha=1.0)),
            'RandomForest':lambda: RandomForestRegressor(n_estimators=150,min_samples_leaf=5,max_features=0.5,n_jobs=1,random_state=0),
            'XGBoost':lambda: XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,n_jobs=1,random_state=0,verbosity=0)}
def predict_all(tr,te):
    pr={'Persistence':te.l0.values,'SeasonalNaive':te.seas.values}
    for n,f in models().items():
        m=f(); m.fit(tr[FEATS],tr.delta); pr[n]=te.l0.values+m.predict(te[FEATS])
    return pr
def rows(fold,h,te,pr,extra=None):
    r=[]
    for n,p in pr.items():
        r.append(pd.DataFrame({'fold':fold,'h':h,'model':n,'state':te.state.values,'wk':te.wk.values,'tdate':te.tdate.values,'y':te.target.values,'yhat':p}))
    return pd.concat(r)
allr=[]
import os
for h in H:
    D=make(h); print('h',h,'rows',len(D),D.wk.min().date(),D.wk.max().date(),flush=True)
    # A: rolling-origin, expanding window, yearly test blocks; training targets strictly precede test origins
    for ty in range(2019,2025):
        t0=pd.Timestamp(f'{ty}-01-01'); t1=pd.Timestamp(f'{ty+1}-01-01')
        tr=D[D.tdate<t0]; te=D[(D.wk>=t0)&(D.wk<t1)]
        allr.append(rows(f'A_{ty}',h,te,predict_all(tr,te)))
    # B: distribution shift: train through 2021, test 2022-2024
    t0=pd.Timestamp('2022-01-01'); tr=D[D.tdate<t0]; te=D[D.wk>=t0]
    allr.append(rows('B_shift',h,te,predict_all(tr,te)))
    # C: leave-one-state-out AND temporal block (train other states <2022, test held-out state 2022-2024)
    for s in sorted(D.state.unique()):
        tr=D[(D.state!=s)&(D.tdate<t0)]; te=D[(D.state==s)&(D.wk>=t0)]
        allr.append(rows('C_LOSO_shift',h,te,predict_all(tr,te)))
    print('done h',h,flush=True)
R=pd.concat(allr,ignore_index=True); R.to_parquet('predictions.parquet') if False else R.to_csv('predictions.csv.gz',index=False)
print('saved',len(R))

"""Supplementary analyses requested by reviewers: regime B by year, regime C per state, bootstrap block lengths, Table I with seasonal-naive, h=2 alert ECE/Brier."""
import pandas as pd, numpy as np
R=pd.concat([pd.read_csv('predictions.csv.gz',parse_dates=['wk','tdate']),pd.read_csv('arima_predictions.csv.gz',parse_dates=['wk','tdate'])],ignore_index=True)
R.to_csv('predictions_all.csv.gz',index=False)
R['ae']=(R.y-R.yhat).abs(); R['cy']=np.expm1(R.y); R['aec']=(R.cy-np.clip(np.expm1(R.yhat),0,None)).abs()
def boot(sub,model,block,B=1000,seed=7):
    rng=np.random.default_rng(seed)
    a=sub[sub.model==model].groupby('wk').ae.agg(['sum','count']); b=sub[sub.model=='Persistence'].groupby('wk').ae.agg(['sum','count'])
    n=len(a); nb=int(np.ceil(n/block)); out=[]
    for _ in range(B):
        st=rng.integers(0,n-block+1,nb); ix=np.concatenate([np.arange(s,s+block) for s in st])[:n]
        out.append(1-a['sum'].values[ix].sum()/a['count'].values[ix].sum()/(b['sum'].values[ix].sum()/b['count'].values[ix].sum()))
    return np.percentile(out,[2.5,97.5])
def stats(g):
    return pd.Series({'MAE_log':g.ae.mean(),'WAPE':g.aec.sum()/g.cy.sum(),'n':len(g)})
MODELS=['Persistence','SeasonalNaive','ARIMA(2,1,1)','Ridge-AR','RandomForest','XGBoost']
# 1. Regime B by test year
B=R[R.fold=='B_shift']; rows=[]
for yr in [2022,2023,2024]:
    S=B[B.wk.dt.year==yr]
    for h in [1,2,4]:
        Sh=S[S.h==h]; pm=Sh[Sh.model=='Persistence'].ae.mean()
        for m in MODELS:
            g=Sh[Sh.model==m]
            if len(g)==0: continue
            d=stats(g).to_dict(); d.update({'year':yr,'h':h,'model':m,'skill':1-d['MAE_log']/pm})
            if m!='Persistence': d['lo'],d['hi']=boot(Sh,m,8)
            rows.append(d)
pd.DataFrame(rows).to_csv('supp_B_by_year.csv',index=False)
# 2. Regime C per state
C=R[R.fold=='C_LOSO_shift']; rows=[]
for (s,h),Sh in C.groupby(['state','h']):
    pm=Sh[Sh.model=='Persistence'].ae.mean()
    for m in MODELS:
        g=Sh[Sh.model==m]
        if len(g): rows.append({'state':s,'h':h,'model':m,'MAE_log':g.ae.mean(),'skill':1-g.ae.mean()/pm,'mean_cases':g.cy.mean(),'n':len(g)})
pd.DataFrame(rows).to_csv('supp_C_by_state.csv',index=False)
# 3. Block-length sensitivity
rows=[]
for setting,S0 in {'A':R[R.fold.str.startswith('A_')],'B':B}.items():
    for h in [1,2,4]:
        Sh=S0[S0.h==h]; pm=Sh[Sh.model=='Persistence'].ae.mean()
        for m in ['Ridge-AR','RandomForest','XGBoost','ARIMA(2,1,1)']:
            sk=1-Sh[Sh.model==m].ae.mean()/pm
            for bl in [1,4,8,13,26]:
                lo,hi=boot(Sh,m,bl); rows.append({'setting':setting,'h':h,'model':m,'block':bl,'skill':sk,'lo':lo,'hi':hi})
pd.DataFrame(rows).to_csv('supp_block_sensitivity.csv',index=False)
print('done')

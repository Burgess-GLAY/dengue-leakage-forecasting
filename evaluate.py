import pandas as pd, numpy as np
rng=np.random.default_rng(7)
R=pd.read_csv('predictions_all.csv.gz',parse_dates=['wk','tdate'])
R['ae']=(R.y-R.yhat).abs(); R['se']=(R.y-R.yhat)**2
R['cnt_y']=np.expm1(R.y); R['cnt_hat']=np.clip(np.expm1(R.yhat),0,None); R['aec']=(R.cnt_y-R.cnt_hat).abs()
def summ(g):
    return pd.Series({'MAE_log':g.ae.mean(),'RMSE_log':np.sqrt(g.se.mean()),'WAPE_count':g.aec.sum()/g.cnt_y.sum(),'n':len(g)})
def bootstrap_skill(sub,model,ref='Persistence',B=1000,block=8):
    """Block bootstrap over weeks: CI of 1 - MAE(model)/MAE(ref)."""
    a=sub[sub.model==model].groupby('wk').ae.agg(['sum','count']); b=sub[sub.model==ref].groupby('wk').ae.agg(['sum','count'])
    wks=a.index.values; n=len(wks); nb=int(np.ceil(n/block)); out=[]
    for _ in range(B):
        st=rng.integers(0,n-block+1,nb); ix=np.concatenate([np.arange(s,s+block) for s in st])[:n]
        out.append(1-a['sum'].values[ix].sum()/a['count'].values[ix].sum()/(b['sum'].values[ix].sum()/b['count'].values[ix].sum()))
    return np.percentile(out,[2.5,97.5])
tab=[]
for (fold,h,model),g in R.groupby(['fold','h','model']): tab.append({'fold':fold,'h':h,'model':model,**summ(g)})
T=pd.DataFrame(tab)
pers=T[T.model=='Persistence'].set_index(['fold','h']).MAE_log
T['skill_vs_persistence']=[1-r.MAE_log/pers[(r.fold,r.h)] for r in T.itertuples()]
T.to_csv('results_by_fold.csv',index=False)
# pooled summaries
def pooled(mask,name):
    S=R[mask]; rows=[]
    for (h,model),g in S.groupby(['h','model']):
        d=summ(g); d['h']=h; d['model']=model; d['setting']=name
        if model!='Persistence':
            lo,hi=bootstrap_skill(S[S.h==h],model); d['skill_lo']=lo; d['skill_hi']=hi
        rows.append(d)
    P=pd.DataFrame(rows); pm=P[P.model=='Persistence'].set_index('h').MAE_log
    P['skill_vs_persistence']=[1-r.MAE_log/pm[r.h] for r in P.itertuples()]; return P
out=pd.concat([pooled(R.fold.str.startswith('A_'),'A rolling-origin 2019-2024'),
               pooled(R.fold=='B_shift','B train<=2021, test 2022-2024'),
               pooled((R.fold=='B_shift')&(R.wk>='2024-01-01'),'B (2024 only)'),
               pooled(R.fold=='C_LOSO_shift','C unseen state + shift 2022-2024')],ignore_index=True)
out.to_csv('results_pooled.csv',index=False)
pd.set_option('display.width',200); pd.set_option('display.float_format','{:.3f}'.format)
print(out[['setting','h','model','MAE_log','RMSE_log','WAPE_count','skill_vs_persistence','skill_lo','skill_hi']].to_string(index=False))

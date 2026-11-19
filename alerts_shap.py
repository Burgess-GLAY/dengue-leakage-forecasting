import pandas as pd, numpy as np, warnings, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier, XGBRegressor
import shap
warnings.filterwarnings('ignore')
P=pd.read_csv('panel_state_week.csv',parse_dates=['wk']).sort_values(['state','wk']).reset_index(drop=True)
P['y']=np.log1p(P.cases)
FEATS=['l0','l1','l2','l3','l4','l5','l6','l7','m4','m8','d1','d4','seas','sin','cos']
GROUPS={'Current level (lags 0-3)':['l0','l1','l2','l3'],'Longer memory (lags 4-7, 4/8-wk means)':['l4','l5','l6','l7','m4','m8'],
        'Momentum (1/4-wk change)':['d1','d4'],'Prior-year same week':['seas'],'Calendar (week-of-year)':['sin','cos']}
def make(h,alert=False):
    out=[]
    for s,g in P.groupby('state'):
        g=g.copy(); y=g.y
        for k in range(8): g[f'l{k}']=y.shift(k)
        g['m4']=y.rolling(4).mean(); g['m8']=y.rolling(8).mean(); g['d1']=y-y.shift(1); g['d4']=y-y.shift(4)
        g['seas']=y.shift(52-h); woy=g.wk.dt.isocalendar().week.astype(int)
        g['sin']=np.sin(2*np.pi*woy/52.18); g['cos']=np.cos(2*np.pi*woy/52.18)
        g['target']=y.shift(-h); g['tdate']=g.wk.shift(-h); g['tcases']=g.cases.shift(-h)
        g['thr']=y.rolling(52,min_periods=52).quantile(0.90)   # state's own trailing 52-week 90th percentile, known at origin
        g['exc']=g.l0-g.thr
        out.append(g)
    D=pd.concat(out).dropna(subset=FEATS+['target']).reset_index(drop=True); D['delta']=D.target-D.l0
    return D
# ---------------- Outbreak-alert task ----------------
AF=FEATS+['thr','exc']
def ece(y,p,bins=10):
    b=np.minimum((p*bins).astype(int),bins-1); e=0
    for i in range(bins):
        m=b==i
        if m.any(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e
res=[]; predrows=[]
for h in [1,2,4]:
    D=make(h).dropna(subset=['thr']).copy()
    D['lab']=((D.target>D.thr)&(D.tcases>=20)).astype(int)
    for fold,(t0,t1) in {**{f'A_{y}':(pd.Timestamp(f'{y}-01-01'),pd.Timestamp(f'{y+1}-01-01')) for y in range(2019,2025)},'B_shift':(pd.Timestamp('2022-01-01'),pd.Timestamp('2025-01-01'))}.items():
        cal0=t0-pd.DateOffset(years=1)
        core=D[D.tdate<cal0]; cal=D[(D.tdate>=cal0)&(D.tdate<t0)]; te=D[(D.wk>=t0)&(D.wk<t1)]
        if te.lab.sum()==0 or core.lab.nunique()<2: continue
        ps={}
        # baseline: logistic on the single origin-time excess over the state's own threshold
        lr=make_pipeline(StandardScaler(),LogisticRegression(C=1.0)).fit(core[['exc']],core.lab)
        ps['Threshold-persistence (LogReg on excess)']=(lr.predict_proba(cal[['exc']])[:,1],lr.predict_proba(te[['exc']])[:,1])
        lr2=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=500)).fit(core[AF],core.lab)
        ps['LogReg (all features)']=(lr2.predict_proba(cal[AF])[:,1],lr2.predict_proba(te[AF])[:,1])
        rf=RandomForestClassifier(n_estimators=150,min_samples_leaf=5,max_features=0.5,random_state=0,n_jobs=1).fit(core[AF],core.lab)
        ps['RandomForest']=(rf.predict_proba(cal[AF])[:,1],rf.predict_proba(te[AF])[:,1])
        xg=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,n_jobs=1,random_state=0,verbosity=0).fit(core[AF],core.lab)
        ps['XGBoost']=(xg.predict_proba(cal[AF])[:,1],xg.predict_proba(te[AF])[:,1])
        for n,(pc,pt) in list(ps.items()):
            if cal.lab.nunique()==2:
                iso=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(pc,cal.lab); ps[n+' + isotonic']=(pc,iso.predict(pt))
        for n,(pc,pt) in ps.items():
            predrows.append(pd.DataFrame({'fold':fold,'h':h,'model':n,'state':te.state.values,'wk':te.wk.values,'lab':te.lab.values,'p':pt}))
R=pd.concat(predrows,ignore_index=True); R.to_csv('alert_predictions.csv.gz',index=False)
def score(g):
    y=g.lab.values; p=g.p.values
    return pd.Series({'prevalence':y.mean(),'PR_AUC':average_precision_score(y,p),'ROC_AUC':roc_auc_score(y,p),'Brier':brier_score_loss(y,p),'ECE':ece(y,p),'n':len(g)})
rows=[]
for name,mask in {'A rolling-origin 2019-2024':R.fold.str.startswith('A_'),'B train<=2021, test 2022-2024':R.fold=='B_shift'}.items():
    for (h,m),g in R[mask].groupby(['h','model']): rows.append({'setting':name,'h':h,'model':m,**score(g)})
AL=pd.DataFrame(rows); AL.to_csv('alert_results.csv',index=False)
pd.set_option('display.width',220); pd.set_option('display.float_format','{:.3f}'.format)
print(AL.to_string(index=False))
# probability-band (tier) table for XGBoost+isotonic, setting B, h=4 and 2
tiers=[]
for h in [2,4]:
    g=R[(R.fold=='B_shift')&(R.h==h)&(R.model=='XGBoost + isotonic')]
    g=g.assign(tier=pd.cut(g.p,[-.01,0.2,0.5,1.01],labels=['Low (<0.2)','Medium (0.2-0.5)','High (>0.5)']))
    t=g.groupby('tier',observed=False).agg(n=('lab','size'),observed_rate=('lab','mean'),mean_pred=('p','mean')).reset_index(); t['h']=h; tiers.append(t)
T=pd.concat(tiers); T.to_csv('alert_tiers.csv',index=False); print(T.to_string(index=False))
# ---------------- SHAP + permutation reliance (XGBoost regressor, setting B) ----------------
sh_rows=[]; perm_rows=[]; yr_rows=[]
for h in [1,4]:
    D=make(h); t0=pd.Timestamp('2022-01-01'); tr=D[D.tdate<t0]; te=D[D.wk>=t0].reset_index(drop=True)
    m=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,n_jobs=1,random_state=0,verbosity=0).fit(tr[FEATS],tr.delta)
    sv=shap.TreeExplainer(m).shap_values(te[FEATS]); S=pd.DataFrame(np.abs(sv),columns=FEATS)
    tot=S.sum(axis=1).mean()
    for gname,cols in GROUPS.items():
        sh_rows.append({'h':h,'group':gname,'mean_abs_shap_group':np.abs(sv[:,[FEATS.index(c) for c in cols]].sum(axis=1)).mean(),'mean_abs_shap_sum_features':S[cols].sum(axis=1).mean()})
        for yr in [2022,2023,2024]:
            mk=(te.wk.dt.year==yr).values
            yr_rows.append({'h':h,'year':yr,'group':gname,'val':np.abs(sv[mk][:,[FEATS.index(c) for c in cols]].sum(axis=1)).mean()})
    # grouped permutation reliance: increase in MAE (delta scale) when group is shuffled jointly across test rows
    base=np.abs(te.delta-m.predict(te[FEATS])).mean(); rng=np.random.default_rng(0)
    for gname,cols in GROUPS.items():
        ds=[]
        for _ in range(20):
            X=te[FEATS].copy(); idx=rng.permutation(len(X)); X[cols]=X[cols].values[idx]; ds.append(np.abs(te.delta-m.predict(X)).mean()-base)
        perm_rows.append({'h':h,'group':gname,'dMAE_mean':np.mean(ds),'dMAE_sd':np.std(ds)})
    if h==4: np.save('shap_values_h4.npy',sv); te.to_csv('shap_test_h4.csv.gz',index=False)
pd.DataFrame(sh_rows).to_csv('shap_groups.csv',index=False); pd.DataFrame(perm_rows).to_csv('perm_groups.csv',index=False); pd.DataFrame(yr_rows).to_csv('shap_groups_by_year.csv',index=False)
print(pd.DataFrame(sh_rows).to_string(index=False)); print(pd.DataFrame(perm_rows).to_string(index=False))

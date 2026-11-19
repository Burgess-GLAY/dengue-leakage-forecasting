import pandas as pd, numpy as np
R=pd.read_csv('predictions_all.csv.gz',parse_dates=['wk','tdate']); R['ae']=(R.y-R.yhat).abs()
# ---- Stable / rising / falling phases (regime B), by horizon
B=R[R.fold=='B_shift']; pers=B[B.model=='Persistence'][['h','state','wk','y','yhat']].rename(columns={'yhat':'l0'}); pers['chg']=pers.y-pers.l0
pers['phase']=pd.cut(pers.chg,[-99,-0.5,0.5,99],labels=['Falling','Stable','Rising'])
M=B[B.model.isin(['Persistence','ARIMA(2,1,1)','Ridge-AR','RandomForest','XGBoost'])].merge(pers[['h','state','wk','phase']],on=['h','state','wk'])
tot=M[M.model=='Persistence'].groupby('h').size()
rows=[]
for (h,ph),g in M.groupby(['h','phase'],observed=True):
    n=len(g[g.model=='Persistence']); d={'h':h,'phase':ph,'n':n,'share_%':100*n/tot[h]}
    for m,gg in g.groupby('model'): d[m]=gg.ae.mean()
    pm=d['Persistence']
    for m in ['RandomForest','XGBoost']: d[m+'_skill']=1-d[m]/pm
    rows.append(d)
ph=pd.DataFrame(rows); ph.to_csv('supp_phase.csv',index=False)
# contribution of stable periods to overall MAE gap (XGBoost vs persistence): share of total absolute error
con=[]
for h,g in M.groupby('h'):
    for m in ['Persistence','XGBoost']:
        x=g[g.model==m]; s=x.groupby('phase',observed=True).ae.sum()/x.ae.sum()
        for p,v in s.items(): con.append({'h':h,'model':m,'phase':p,'share_of_total_abs_error_%':100*v})
pd.DataFrame(con).to_csv('supp_phase_error_share.csv',index=False)
# ---- Alert operating points (regime B)
A=pd.read_csv('alert_predictions.csv.gz',parse_dates=['wk']); A=A[A.fold=='B_shift']
rows=[]
for (h,m),g in A.groupby(['h','model']):
    if m not in ['XGBoost','RandomForest','Threshold-persistence (LogReg on excess)']: continue
    y=g.lab.values
    for th in [0.1,0.2,0.3,0.5]:
        p=g.p.values>=th; tp=(p&(y==1)).sum(); fp=(p&(y==0)).sum(); fn=((~p)&(y==1)).sum(); tn=((~p)&(y==0)).sum()
        rows.append({'h':h,'model':m,'threshold':th,'prevalence':y.mean(),'alerts_issued':int(p.sum()),'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),
                     'false_alarm_ratio(FP/alerts)':fp/max(tp+fp,1),'false_positive_rate(FP/negatives)':fp/max(fp+tn,1),'TP':tp,'FP':fp,'FN':fn,'TN':tn})
op=pd.DataFrame(rows); op.to_csv('supp_alert_operating.csv',index=False)
# prevalence by year
pv=A[(A.model=='XGBoost')].assign(year=lambda d:d.wk.dt.year).groupby(['h','year']).lab.agg(['mean','sum','size']).reset_index(); pv.columns=['h','year','prevalence','n_alerts_true','n']; pv.to_csv('supp_alert_prevalence_by_year.csv',index=False)
# threshold evolution: state trailing-52w 90th percentile in weekly cases
P=pd.read_csv('panel_state_week.csv',parse_dates=['wk']).sort_values(['state','wk']); P['y']=np.log1p(P.cases)
P['thr']=P.groupby('state').y.transform(lambda s:s.rolling(52,min_periods=52).quantile(0.9)); P['thr_cases']=np.expm1(P.thr)
P=P.dropna(subset=['thr']); th=P.groupby(P.wk.dt.year).thr_cases.agg(['median',lambda s:s.quantile(.25),lambda s:s.quantile(.75)]).reset_index(); th.columns=['year','median_cases','p25','p75']
th.to_csv('supp_threshold_by_year.csv',index=False)
pd.set_option('display.width',220)
print(ph.round(3).to_string(index=False)); print(pd.DataFrame(con).round(1).to_string(index=False)); print(op[op.model=='XGBoost'].round(3).to_string(index=False)); print(pv.round(3).to_string(index=False)); print(th.round(1).to_string(index=False))

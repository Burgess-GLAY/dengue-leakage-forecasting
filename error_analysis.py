import pandas as pd, numpy as np
R=pd.read_csv('predictions_all.csv.gz',parse_dates=['wk','tdate'])
P=pd.read_csv('panel_state_week.csv',parse_dates=['wk'])
size=P[P.wk<'2022-01-01'].groupby('state').cases.mean()
q=pd.qcut(size,4,labels=['Q1 (smallest)','Q2','Q3','Q4 (largest)']); R['size']=R.state.map(q)
B=R[(R.fold=='B_shift')&(R.h==4)&R.model.isin(['Persistence','XGBoost','RandomForest','ARIMA(2,1,1)'])].copy()
l0=R[(R.fold=='B_shift')&(R.h==4)&(R.model=='Persistence')].set_index(['state','wk']).yhat  # persistence prediction == y at origin
B['l0']=[l0.loc[(s,w)] for s,w in zip(B.state,B.wk)]
B['chg']=B.y-B.l0; B['phase']=pd.cut(B.chg,[-99,-0.5,0.5,99],labels=['Falling (<-0.5)','Stable (+-0.5)','Rising (>+0.5)'])
B['ae']=(B.y-B.yhat).abs()
a=B.pivot_table(index='size',columns='model',values='ae',aggfunc='mean',observed=True)
b=B.pivot_table(index='phase',columns='model',values='ae',aggfunc='mean',observed=True)
cnt=B[B.model=='Persistence'].groupby('phase',observed=True).size()
a.to_csv('err_by_size.csv'); b.assign(n=cnt).to_csv('err_by_phase.csv')
print(a.round(3).to_string()); print(b.assign(n=cnt).round(3).to_string())
print('states per size tier:',q.value_counts().to_dict())

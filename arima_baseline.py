"""Per-state ARIMA(2,1,1) baseline on log1p cases. Parameters fitted once per (state, fold) on data strictly before the test block;
then the model state is updated with newly observed weeks (parameters fixed) and h-step forecasts are issued from every origin,
using only data up to that origin (dynamic prediction)."""
import pandas as pd, numpy as np, warnings, time, sys
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings('ignore')
P=pd.read_csv('panel_state_week.csv',parse_dates=['wk']).sort_values(['state','wk'])
P['y']=np.log1p(P.cases)
HS=[1,2,4]
folds={f'A_{y}':(pd.Timestamp(f'{y}-01-01'),pd.Timestamp(f'{y+1}-01-01')) for y in range(2019,2025)}
folds['B_shift']=(pd.Timestamp('2022-01-01'),pd.Timestamp('2025-01-01'))
only=sys.argv[1:] 
out=[]; t_start=time.time()
for fold,(t0,t1) in folds.items():
    if only and fold not in only: continue
    for s,g in P.groupby('state'):
        g=g.reset_index(drop=True); y=g.y.values; wk=g.wk
        i0=int((wk<t0).sum()); i1=int((wk<t1).sum())
        try:
            res=ARIMA(y[:i0],order=(2,1,1)).fit()
            ext=res.extend(y[i0:i1]) if False else res.extend(y[i0:])
        except Exception as e:
            print('fail',fold,s,e); continue
        # origins j in [i0,i1): forecast j+1..j+4 using data up to j (dynamic from j+1)
        for j in range(i0,i1):
            if j+max(HS)>=len(y): continue
            pr=ext.predict(start=j+1-i0,end=j+max(HS)-i0,dynamic=0)
            for h in HS:
                out.append((fold,h,s,wk[j],wk[j+h],y[j+h],pr[h-1]))
    print(fold,'done',round(time.time()-t_start),'s',flush=True)
A=pd.DataFrame(out,columns=['fold','h','state','wk','tdate','y','yhat']); A['model']='ARIMA(2,1,1)'
A.to_csv('arima_predictions.csv.gz',index=False); print(len(A))

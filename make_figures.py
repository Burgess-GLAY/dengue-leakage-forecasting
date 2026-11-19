import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import precision_recall_curve, average_precision_score
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight'})
C={'Persistence':'#7f7f7f','ARIMA(2,1,1)':'#8c564b','Ridge-AR':'#1f77b4','RandomForest':'#2ca02c','XGBoost':'#d62728'}
def save(fig,n): fig.savefig(f'figs/{n}.png'); fig.savefig(f'figs/{n}.pdf'); plt.close(fig)
P=pd.read_csv('panel_state_week.csv',parse_dates=['wk'])
# ---- Fig 2: data overview
fig,ax=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={'width_ratios':[1,1.25]})
nat=P.groupby('wk').cases.sum()/1e3
ax[0].plot(nat.index,nat.values,color='#1f4e79',lw=0.9); ax[0].set_ylabel('Probable cases per week (thousands)'); ax[0].set_title('(a) National weekly series (26 states)')
ax[0].set_xlabel('Year')
M=P.pivot(index='state',columns='wk',values='cases'); M=M.loc[M.sum(axis=1).sort_values(ascending=False).index]
im=ax[1].imshow(np.log1p(M.values),aspect='auto',cmap='viridis',extent=[0,M.shape[1],M.shape[0],0])
yrs=pd.to_datetime(M.columns); ticks=[i for i in range(len(yrs)) if yrs[i].month==1 and yrs[i].day<=7]
ax[1].set_xticks(ticks); ax[1].set_xticklabels([yrs[i].year for i in ticks],rotation=0); ax[1].set_yticks([])
ax[1].set_ylabel('State (ordered by total cases)'); ax[1].set_title('(b) State-week panel, log(1+cases)'); cb=fig.colorbar(im,ax=ax[1],pad=0.02); cb.set_label('log(1+cases)')
save(fig,'fig2_data_overview')
# ---- Fig 3: skill vs persistence
pl=pd.read_csv('results_pooled.csv'); sets=[('A rolling-origin 2019-2024','(a) Rolling-origin, 2019–2024'),('B train<=2021, test 2022-2024','(b) Distribution shift, test 2022–2024'),('C unseen state + shift 2022-2024','(c) Unseen state + shift')]
order=['ARIMA(2,1,1)','Ridge-AR','RandomForest','XGBoost']
fig,axs=plt.subplots(1,3,figsize=(11,3.4),sharey=True)
for ax,(sk,tt) in zip(axs,sets):
    d=pl[pl.setting==sk]; w=0.2
    for i,m in enumerate(order):
        g=d[d.model==m].sort_values('h')
        if g.empty: continue
        x=np.arange(3)+(i-1.5)*w
        ax.bar(x,g.skill_vs_persistence*100,w,color=C[m],label=m,yerr=[(g.skill_vs_persistence-g.skill_lo)*100,(g.skill_hi-g.skill_vs_persistence)*100],capsize=2,error_kw={'lw':0.8})
    ax.axhline(0,color='k',lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(['h = 1','h = 2','h = 4']); ax.set_title(tt)
axs[0].set_ylabel('MAE reduction vs. persistence (%)'); axs[0].legend(loc='upper left',frameon=False)
fig.text(0.5,-0.04,'Error bars: 95% moving-block bootstrap intervals over weeks. Seasonal-naïve (skill < −65%) omitted for scale; ARIMA is per-state, so not defined in (c).',ha='center',fontsize=7.5)
save(fig,'fig3_skill_vs_persistence')
# ---- Fig 4: example forecasts (setting B, h=4)
R=pd.read_csv('predictions_all.csv.gz',parse_dates=['wk','tdate']); B=R[(R.fold=='B_shift')&(R.h==4)]
fig,axs=plt.subplots(2,2,figsize=(10,5.2),sharex=True)
for ax,s in zip(axs.ravel(),['SAO PAULO','MINAS GERAIS','PARANA','GOIAS']):
    for m,ls,lw in [('Persistence',':',1.0),('XGBoost','-',1.3)]:
        g=B[(B.state==s)&(B.model==m)].sort_values('tdate')
        if m=='XGBoost': ax.plot(g.tdate,np.expm1(g.y)/1e3,color='k',lw=1.4,label='Observed')
        ax.plot(g.tdate,np.clip(np.expm1(g.yhat),0,None)/1e3,ls=ls,color=C[m],lw=lw,label=f'{m} (4-week-ahead)')
    ax.set_title(s.title()); ax.set_ylabel('Cases per week (thousands)')
axs[0,0].legend(frameon=False); fig.autofmt_xdate(rotation=0,ha='center'); save(fig,'fig4_example_forecasts')
# ---- Fig 5: explainability
sg=pd.read_csv('shap_groups.csv'); pg=pd.read_csv('perm_groups.csv'); yr=pd.read_csv('shap_groups_by_year.csv')
gs=list(sg[sg.h==1].group); short=['Current\nlevel','Longer\nmemory','Momentum','Prior-year\nsame week','Calendar']
fig,axs=plt.subplots(1,3,figsize=(11.5,3.6))
for j,(df,col,tt,yl) in enumerate([(sg,'mean_abs_shap_group','(a) Grouped mean |SHAP|','Mean |SHAP| of predicted log-change'),(pg,'dMAE_mean','(b) Grouped permutation reliance','Increase in MAE when group is permuted')]):
    for k,(h,c) in enumerate([(1,'#9ecae1'),(4,'#08519c')]):
        d=df[df.h==h].set_index('group').loc[gs]; axs[j].bar(np.arange(5)+(k-0.5)*0.38,d[col],0.38,color=c,label=f'h = {h}',yerr=d['dMAE_sd'] if col=='dMAE_mean' else None,capsize=2)
    axs[j].set_xticks(range(5)); axs[j].set_xticklabels(short,fontsize=8); axs[j].set_title(tt); axs[j].set_ylabel(yl)
axs[0].legend(frameon=False)
y4=yr[yr.h==4].pivot(index='group',columns='year',values='val').loc[gs]
for k,(yv,c) in enumerate(zip([2022,2023,2024],['#c7e9c0','#41ab5d','#00441b'])): axs[2].bar(np.arange(5)+(k-1)*0.27,y4[yv],0.27,color=c,label=str(yv))
axs[2].set_xticks(range(5)); axs[2].set_xticklabels(short,fontsize=8); axs[2].set_title('(c) Grouped mean |SHAP| by test year (h = 4)'); axs[2].legend(frameon=False,title='Test year',title_fontsize=7)
save(fig,'fig5_explainability')
# ---- Fig 6: alerts
A=pd.read_csv('alert_predictions.csv.gz'); Ab=A[(A.fold=='B_shift')&(A.h==4)]
fig,axs=plt.subplots(1,2,figsize=(9,3.6))
cm={'XGBoost':'#d62728','RandomForest':'#2ca02c','LogReg (all features)':'#1f77b4','Threshold-persistence (LogReg on excess)':'#7f7f7f'}
for m,c in cm.items():
    g=Ab[Ab.model==m]; p,r,_=precision_recall_curve(g.lab,g.p); axs[0].plot(r,p,color=c,lw=1.4,label=f'{m.split(" (")[0]} (AP {average_precision_score(g.lab,g.p):.2f})')
axs[0].axhline(Ab.lab.mean(),color='k',ls=':',lw=0.8); axs[0].text(0.02,Ab.lab.mean()+0.02,'prevalence',fontsize=7)
axs[0].set_xlabel('Recall'); axs[0].set_ylabel('Precision'); axs[0].set_title('(a) Precision–recall, 4-week-ahead alert'); axs[0].legend(frameon=False,loc='lower left')
for m,c,l in [('XGBoost','#d62728','XGBoost (uncalibrated)'),('XGBoost + isotonic','#ff9896','XGBoost + isotonic')]:
    g=Ab[Ab.model==m]; b=pd.cut(g.p,np.linspace(0,1,11),include_lowest=True); t=g.groupby(b,observed=True).agg(p=('p','mean'),o=('lab','mean'),n=('lab','size'))
    t=t[t.n>=15]; axs[1].plot(t.p,t.o,'o-',color=c,ms=4,lw=1.2,label=l)
axs[1].plot([0,1],[0,1],'k:',lw=0.8); axs[1].set_xlabel('Predicted probability'); axs[1].set_ylabel('Observed alert frequency'); axs[1].set_title('(b) Reliability (bins with n ≥ 15)'); axs[1].legend(frameon=False)
save(fig,'fig6_alerts')
# ---- Fig 7: error analysis
a=pd.read_csv('err_by_size.csv',index_col=0); b=pd.read_csv('err_by_phase.csv',index_col=0)
fig,axs=plt.subplots(1,2,figsize=(9.5,3.4),sharey=True)
for ax,df,tt,xl in [(axs[0],a,'(a) By state size (mean training-period cases)',None),(axs[1],b.drop(columns='n'),'(b) By realised 4-week change in log cases',None)]:
    for i,m in enumerate(['Persistence','ARIMA(2,1,1)','RandomForest','XGBoost']):
        ax.bar(np.arange(len(df))+(i-1.5)*0.2,df[m],0.2,color=C[m],label=m)
    ax.set_xticks(range(len(df))); ax.set_xticklabels([str(i) for i in df.index],fontsize=8); ax.set_title(tt)
axs[0].set_ylabel('MAE (log scale), setting B, h = 4'); axs[1].legend(frameon=False,fontsize=7)
save(fig,'fig7_error_analysis')
# ---- Fig 1: architecture (implemented pipeline only)
fig,ax=plt.subplots(figsize=(11,5.6)); ax.set_xlim(0,11); ax.set_ylim(0,5.6); ax.axis('off')
def box(x,y,w,h,t,fc='#e8eef7',ec='#1f4e79',fs=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.08',fc=fc,ec=ec,lw=1.1)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def arr(x1,y1,x2,y2,ls='-'): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,lw=1.1,color='#333',linestyle=ls))
box(0.1,4.0,2.2,1.2,'Data source\nOpenDengue V1.3\nTemporal extract\n(public GitHub release)')
box(2.8,4.0,2.5,1.2,'Filtering and harmonisation\nBrazil, weekly, MOH municipal,\nprobable cases; WHO/PAHO\nnational rows removed')
box(5.8,4.0,2.3,1.2,'State-week panel\n26 states x 573 weeks\n(2014-2024)\nlog(1 + cases)')
box(8.6,4.0,2.3,1.2,'Leakage-safe features\nlags 0-7, rolling means,\nmomentum, prior-year value,\nweek-of-year encoding')
arr(2.3,4.6,2.8,4.6); arr(5.3,4.6,5.8,4.6); arr(8.1,4.6,8.6,4.6)
box(0.1,2.1,3.6,1.3,'Forecasting models (h = 1, 2, 4)\nBaselines: persistence, seasonal naive,\nper-state ARIMA(2,1,1), ridge AR\nTree ensembles: Random Forest, XGBoost',fc='#eaf5ea',ec='#2e7d32')
box(4.1,2.1,3.0,1.3,'Outbreak-alert classifier\nlabel: target week above the state\'s\ntrailing 52-week 90th percentile;\nlogistic, Random Forest, XGBoost',fc='#eaf5ea',ec='#2e7d32')
box(7.6,2.1,3.3,1.3,'Validation settings\nA: rolling-origin (2019-2024)\nB: train <= 2021, test 2022-2024\nC: unseen state + temporal shift',fc='#fdf3e3',ec='#b26a00')
arr(9.75,4.0,5.6,3.4); arr(9.3,4.0,1.9,3.4)
arr(7.6,2.75,7.1,2.75,ls='--'); ax.text(7.35,2.9,'splits',fontsize=6.5,ha='center',color='#b26a00')
box(0.1,0.15,5.0,1.1,'Evaluation\nMAE / RMSE, skill vs. persistence (block-bootstrap CIs)\nPR-AUC, Brier score, calibration',fc='#f3e8f7',ec='#6a1b9a')
box(5.5,0.15,5.4,1.1,'Explainability and error analysis\ngrouped SHAP and permutation reliance (XGBoost regressor)\nerrors by state size and epidemic phase',fc='#f3e8f7',ec='#6a1b9a')
arr(1.9,2.1,1.9,1.25); arr(5.6,2.1,3.6,1.25); arr(2.8,2.1,7.0,1.25)
ax.text(5.5,5.42,'Offline batch pipeline in Python (pandas, scikit-learn, XGBoost, statsmodels, SHAP); no distributed or streaming component',ha='center',fontsize=8,style='italic',color='#444')
save(fig,'fig1_architecture')
print('ok')

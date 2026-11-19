"""Build the Brazil state-week dengue panel from OpenDengue V1.3 (MOH-BRA municipal weekly rows only)."""
import pandas as pd, numpy as np
SRC='/home/claude/od/data/releases/V1.3/x/Temporal_extract_V1_3.csv'
cols=['adm_0_name','adm_1_name','adm_2_name','calendar_start_date','dengue_total','case_definition_standardised','S_res','T_res','UUID']
d=pd.read_csv(SRC,usecols=cols,low_memory=False)
b=d[(d.adm_0_name=='BRAZIL')&(d.T_res=='Week')]
n0=len(b)
# exclude WHO/PAHO national rows (adm_2 missing) to avoid double counting the same cases
m=b[b.UUID.str.startswith('MOH-BRA')&b.adm_2_name.notna()&(b.case_definition_standardised=='Probable')].copy()
print('rows Brazil weekly',n0,'-> municipal MOH probable',len(m))
m['wk']=pd.to_datetime(m.calendar_start_date)
s=m.groupby(['adm_1_name','wk'],as_index=False).dengue_total.sum()
last=s.groupby('adm_1_name').wk.max()
print('states with series ending before 2024:\n',last[last<'2024-12-01'])
keep=last[last>='2024-12-01'].index
s=s[s.adm_1_name.isin(keep)]
grid=pd.date_range('2014-01-05','2024-12-22',freq='7D')
idx=pd.MultiIndex.from_product([sorted(keep),grid],names=['state','wk'])
p=s.rename(columns={'adm_1_name':'state'}).set_index(['state','wk']).dengue_total.reindex(idx)
print('implicit zero-filled state-weeks:',int(p.isna().sum()),'of',len(p))
p=p.fillna(0.0).rename('cases').reset_index()
p.to_csv('panel_state_week.csv',index=False)
print(p.groupby(p.wk.dt.year).cases.sum().astype(int).to_string())
print('states',p.state.nunique(),'weeks',p.wk.nunique(),'rows',len(p))

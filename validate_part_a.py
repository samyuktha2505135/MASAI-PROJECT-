"""Assert deterministic Part A reference checkpoints."""
import sqlite3
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
raw = ROOT/'data/raw'
clean = ROOT/'data/clean'
o = pd.read_csv(raw/'orders.csv'); c=pd.read_csv(raw/'customers.csv')
assert len(o)==13084
assert o.duplicated().sum()==119
assert o.status.eq('pending').sum()==656
assert o.order_id.nunique()==12309
assert pd.to_numeric(o.order_amount,errors='coerce').isna().sum()==120
assert o.delivery_minutes.lt(0).sum()==106
assert o.delivery_minutes.isna().sum()==2541
assert c.age.isna().sum()==61
assert (~c.age.between(10,100) & c.age.notna()).sum()==6
assert c.city.nunique()==24
assert c.gender.nunique()==10
assert c.gender.isna().sum()==30
oc=pd.read_csv(clean/'orders.csv');cc=pd.read_csv(clean/'customers.csv')
assert len(oc)==12309 and oc.order_id.nunique()==12309
assert oc.status.eq('completed').sum()==11110
assert oc.status.eq('cancelled').sum()==1199
assert not oc.status.eq('pending').any()
assert oc.order_amount.notna().all()
assert oc.loc[oc.status=='completed','delivery_minutes'].notna().all()
assert oc.loc[oc.status=='cancelled','delivery_minutes'].isna().sum()==1199
assert abs(oc.loc[oc.status=='completed','order_amount'].sum()-2998668.80)<.001
assert cc.city.value_counts().to_dict()==dict(Pune=227,Delhi=210,Bengaluru=197,Chennai=191,Hyderabad=191,Mumbai=184)
assert cc.gender.value_counts().to_dict()==dict(female=610,male=560,unknown=30)
assert cc.age.notna().all() and cc.age.between(18,60).all()
with sqlite3.connect(ROOT/'data/instamart.db') as db:
    assert {'customers','products','orders'} <= set(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'",db)['name'])
print('ALL CHECKPOINTS PASSED')

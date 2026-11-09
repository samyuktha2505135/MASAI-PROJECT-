"""Cross-part and assessment-artifact assertions."""
import json,sqlite3
from pathlib import Path
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parent
clean=ROOT/'data/clean';outputs=ROOT/'outputs';reports=ROOT/'reports'
customers=pd.read_csv(clean/'customers.csv')
products=pd.read_csv(clean/'products.csv')
orders=pd.read_csv(clean/'orders.csv')
assert len(customers)==1200 and len(products)==40 and len(orders)==12309
assert len(orders)==orders.order_id.nunique()
assert (orders.status=='completed').sum()==11110 and (orders.status=='cancelled').sum()==1199
assert round(orders.loc[orders.status.eq('completed'),'order_amount'].sum(),2)==2998668.80
assert orders.loc[orders.status.eq('completed'),'delivery_minutes'].notna().all()
assert orders.loc[orders.status.eq('cancelled'),'delivery_minutes'].isna().all()
with sqlite3.connect(ROOT/'data/instamart.db') as con:
    assert {'customers','products','orders'}.issubset({x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table'")})
    assert con.execute('SELECT COUNT(*) FROM orders').fetchone()[0]==12309
assert (ROOT/'sql/queries.sql').read_text().count(';')>=7
segments=pd.read_csv(outputs/'customer_segments.csv')
assert len(segments)==1175 and segments.customer_id.is_unique
assert {'customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium'}.issubset(segments.columns)
profile=segments.groupby('segment_id').monetary.mean()
assert segments.loc[segments.is_premium,'segment_id'].nunique()==1
assert int(segments.loc[segments.is_premium,'segment_id'].iloc[0])==int(profile.idxmax())
assert (outputs/'lapse_model.joblib').is_file()
model=joblib.load(outputs/'lapse_model.joblib');assert callable(model.predict)
assert {'customer_id','lapse_probability','actually_lapsed'}.issubset(pd.read_csv(outputs/'retention_list.csv').columns)
triage=pd.read_csv(outputs/'triage_results.csv',keep_default_na=False)
assert len(triage)==164 and len(pd.read_csv(outputs/'triage_budget_demo.csv'))==164
assert (triage.route.eq('security_team') & triage.category.ne('fraud')).sum()==0
assert triage.loc[triage.status.isin(['skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted']),'route'].eq('human_review').all()
assert triage.status.value_counts().to_dict()=={'classified_json':135,'classified_keyword':12,'flagged_invalid_output':9,'flagged_long':4,'skipped_empty':4}
b=json.loads((reports/'part_b_results.json').read_text());assert b['cohort']==1156 and b['lapsed']==443
assert max(b['logistic']['test']['f1'],b['forest']['test']['f1'])>=.55
c=json.loads((reports/'part_c_results.json').read_text());assert c['redactions']['total']==278 and c['few_shot']['accuracy']==1
assert c['zero_shot']['accuracy']>=.852 and c['budget_demo']['exhausted']>=1
d=json.loads((reports/'part_d_results.json').read_text())
assert d['chunk_count']==15 and d['policy_eval_overlap_10']=={'retrieval_correct':12,'fact_correct':12,'unanswerable_fallback':2}
assert [x['status'] for x in d['agent_runs']]==['REFUND','REFUND','ESCALATE','DENIED','DENIED','NOT_FOUND','FALLBACK_MAX_ITERATIONS']
assert d['token_limit_demo']=='FALLBACK_TOKEN_LIMIT' and d['timeout_demo']=='FALLBACK_TIMEOUT'
assert d['recovery']=={'status':'REFUND','tools':['get_order','search_policy','calculate_refund','issue_refund']}
lines=[json.loads(x) for x in (outputs/'agent_runs.jsonl').read_text().splitlines() if x.strip()]
assert len(lines)==7 and len({x['scenario_id'] for x in lines})==7
assert (outputs/'refund_ledger.jsonl').exists() and (ROOT/'state').is_dir()
assert (ROOT/'generate_data.py').exists() and (ROOT/'mock_llm.py').exists()
print('ALL INTEGRATION CHECKS PASSED')

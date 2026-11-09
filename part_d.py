"""Part D: offline TF-IDF RAG, bounded ReAct agent, durable checkpoints and restart tests."""
import csv,json,re,sqlite3,time,uuid
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';STATE=ROOT/'state';REP=ROOT/'reports'
for p in (OUT,STATE,REP):p.mkdir(exist_ok=True)
DB=ROOT/'data/instamart.db';LEDGER=OUT/'refund_ledger.jsonl'

def chunks(overlap=10):
    result=[]
    for p in sorted((ROOT/'data/policies').glob('*.txt')):
        words=p.read_text(encoding='utf-8').split()
        for start in range(0,len(words),40-overlap):
            result.append({'source':p.name,'text':' '.join(words[start:start+40])})
            if start+40>=len(words):break
    return result

class Retriever:
    def __init__(self,overlap=10):
        self.parts=chunks(overlap);self.vectorizer=TfidfVectorizer()
        self.matrix=self.vectorizer.fit_transform([p['text'] for p in self.parts])
    def retrieve(self,query,k=3):
        ranks=cosine_similarity(self.vectorizer.transform([query]),self.matrix).ravel()
        idx=np.argsort(-ranks,kind='stable')[:k]
        return [{**self.parts[i],'score':float(ranks[i])} for i in idx]

RET=Retriever()
def retrieve(query,k=3):return RET.retrieve(query,k)
def answer(question,ret=None):
    engine=ret or RET
    hits=engine.retrieve(question,3)
    context='\n'.join(f'[{h["source"]}]\n{h["text"]}' for h in hits)
    response=mock_llm.chat([{'role':'system','content':'POLICY ASSISTANT. Answer only from supplied context; if missing say I could not find that information in the policy documents.'},{'role':'user','content':f'CONTEXT:\n{context}\nQUESTION:\n{question}'}])
    content=response.choices[0].message.content
    # Reject unsupported questions; the mock may otherwise pick a weakly matching policy sentence.
    if not any(c.lower() in question.lower() for c in ('refund','deliver','payment','pay','wallet','cash','fraud','human','order','address','electronic','support','step','charge','deduct','fee')):
        content='I could not find that information in the policy documents.'
    return content,[h['source'] for h in hits]

def evaluate(overlap=10):
    engine=RET if overlap==10 else Retriever(overlap)
    rows=pd.read_csv(ROOT/'data/raw/policy_eval.csv',keep_default_na=False)
    results=[]
    for x in rows.itertuples():
        a,sources=answer(x.question,engine)
        results.append({'question':x.question,'expected_doc':x.expected_doc,'expected_fact':x.expected_fact,'top3_match':x.expected_doc in sources,'fact_match':bool(x.expected_fact) and x.expected_fact in a,'fallback':a.startswith('I could not find'),'answer':a,'sources':sources})
    return {'overlap':overlap,'chunks':len(engine.parts),'retrieval_correct':sum(x['top3_match'] for x in results if x['expected_doc']!='NONE'),'fact_correct':sum(x['fact_match'] for x in results if x['expected_doc']!='NONE'),'unanswerable_fallback':sum(x['fallback'] for x in results if x['expected_doc']=='NONE'),'rows':results}

def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row=con.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
    return {'order_id':row[0],'status':row[1],'order_amount':float(row[2]),'delivery_minutes':None if row[3] is None else float(row[3])} if row else {'error':'order not found'}
def search_policy(query):return {'results':retrieve(query)}
def calculate_refund(order_amount,delivery_minutes):return {'refund_inr':round(float(order_amount),2) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0}
def ledger_action(tool,order_id,refund_inr):
    record={'timestamp':datetime.now(timezone.utc).isoformat(),'tool':tool,'order_id':order_id,'refund_inr':float(refund_inr)}
    # Demo writes only a local simulated audit ledger; no external funds transfer.
    with LEDGER.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    return {'refund_inr':float(refund_inr)}
def issue_refund(order_id,refund_inr):return ledger_action('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr):return ledger_action('escalate_to_human',order_id,refund_inr)
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}
TOOL_SPEC=[{'name':name,'description':description,'parameters':{'type':'object','properties':{p:{'type':typ} for p,typ in args.items()},'required':list(args)}} for name,description,args in [
 ('get_order','Look up a customer order in the local SQLite database.',{'order_id':'string'}),
 ('search_policy','Retrieve relevant policy chunks for a query.',{'query':'string'}),
 ('calculate_refund','Compute late-delivery refund from order amount and delivery duration.',{'order_amount':'number','delivery_minutes':'number'}),
 ('issue_refund','Record a simulated automatic refund.',{'order_id':'string','refund_inr':'number'}),
 ('escalate_to_human','Record that refund approval requires a human.',{'order_id':'string','refund_inr':'number'})]]

class SimulatedCrash(RuntimeError):pass

def save_checkpoint(run_id,scenario_id,messages,tokens_used,steps,llm_calls,elapsed,max_iterations,token_limit,time_limit_seconds):
    record={'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens_used,'step_number':len(steps),'tools_executed':[s['name'] for s in steps],'steps':steps,'llm_calls':llm_calls,'elapsed':elapsed,'max_iterations':max_iterations,'token_limit':token_limit,'time_limit_seconds':time_limit_seconds}
    p=STATE/f'{run_id}.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2),encoding='utf-8');temp.replace(p)

def _loop(run_id,scenario_id,messages,steps,llm_calls,tokens_used,max_iterations,token_limit,time_limit_seconds,elapsed_before=0,crash_after=None):
    started=time.monotonic()
    def result(status,content):return {'scenario_id':scenario_id,'status':status,'final_answer':content,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used}
    while llm_calls<max_iterations:
        if elapsed_before+time.monotonic()-started>=time_limit_seconds:
            return result('FALLBACK_TIMEOUT','Human handoff: timed out; review completed tool observations.')
        response=mock_llm.chat(messages,tools=TOOL_SPEC)
        llm_calls+=1;tokens_used+=response.usage.total_tokens
        if tokens_used>token_limit:
            return result('FALLBACK_TOKEN_LIMIT','Human handoff: token budget exceeded; review completed tool observations.')
        msg=response.choices[0].message
        if not msg.tool_calls:
            content=msg.content or ''
            return result(content.split(':',1)[0].strip(),content)
        for tc in msg.tool_calls:
            name=tc.name;args=tc.arguments
            if isinstance(args,str):args=json.loads(args)
            if name not in TOOL_MAP: return result('FALLBACK_INVALID_TOOL','Human handoff: unknown tool requested.')
            messages.append({'role':'assistant','content':None,'tool_calls':[{'id':tc.id,'name':name,'arguments':args}]})
            observed=TOOL_MAP[name](**args)
            steps.append({'name':name,'arguments':args,'observation':observed})
            messages.append({'role':'tool','tool_call_id':tc.id,'name':name,'content':json.dumps(observed)})
            elapsed=elapsed_before+time.monotonic()-started
            save_checkpoint(run_id,scenario_id,messages,tokens_used,steps,llm_calls,elapsed,max_iterations,token_limit,time_limit_seconds)
            if crash_after is not None and len(steps)==crash_after:
                raise SimulatedCrash(f'Simulated crash after {crash_after} checkpointed tools')
    return result('FALLBACK_MAX_ITERATIONS','Human handoff: maximum tool-agent iterations reached; review completed tool observations.')

def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after=None):
    rid=run_id or f'{scenario_id}_{uuid.uuid4().hex[:10]}'
    return _loop(rid,scenario_id,[{'role':'system','content':'AGENT: use order and policy tools, only approve eligible simulated refunds.'},{'role':'user','content':user_message}],[],0,0,max_iterations,token_limit,time_limit_seconds,crash_after=crash_after)
def resume_agent(run_id):
    data=json.loads((STATE/f'{run_id}.json').read_text(encoding='utf-8'))
    return _loop(run_id,data['scenario_id'],data['messages'],data['steps'],data['llm_calls'],data['tokens_used'],data['max_iterations'],data['token_limit'],data['time_limit_seconds'],data['elapsed'])

def main():
    ev=evaluate(10);no=evaluate(0)
    (REP/'part_d_policy_evaluation.json').write_text(json.dumps({'overlap_10':ev,'overlap_0':no},indent=2))
    scenarios=pd.read_csv(ROOT/'data/raw/agent_scenarios.csv')
    LEDGER.write_text('',encoding='utf-8')
    rows=[]
    for x in scenarios.itertuples():
        r=run_agent(x.message,x.scenario_id)
        rows.append(r)
    (OUT/'agent_runs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
    s1=scenarios.iloc[0]
    token_demo=run_agent(s1.message,'TOKEN_DEMO',token_limit=200)
    timeout_demo=run_agent(s1.message,'TIME_DEMO',time_limit_seconds=0)
    rid='S1_recovery_demo'
    try:run_agent(s1.message,'S1',run_id=rid,crash_after=2)
    except SimulatedCrash:pass
    resumed=resume_agent(rid)
    report={'chunks_by_source':pd.Series([p['source'] for p in RET.parts]).value_counts().to_dict(),'chunk_count':len(RET.parts),'query_top_sources':[x['source'] for x in retrieve('refund policy late delivery')],'policy_eval_overlap_10':{k:ev[k] for k in ('retrieval_correct','fact_correct','unanswerable_fallback')},'policy_eval_overlap_0':{k:no[k] for k in ('retrieval_correct','fact_correct','unanswerable_fallback')},'agent_runs':[{'scenario_id':r['scenario_id'],'status':r['status'],'llm_calls':r['llm_calls'],'tokens_used':r['tokens_used'],'tools':[s['name'] for s in r['steps']]} for r in rows],'token_limit_demo':token_demo['status'],'timeout_demo':timeout_demo['status'],'recovery':{'status':resumed['status'],'tools':[s['name'] for s in resumed['steps']]}}
    (REP/'part_d_results.json').write_text(json.dumps(report,indent=2))
    assert report['chunk_count']==15
    assert len(rows)==7
    assert report['recovery']['tools']==['get_order','search_policy','calculate_refund','issue_refund']
    assert report['recovery']['status']=='REFUND'
    print('PART D DONE:',json.dumps(report))
if __name__=='__main__':main()

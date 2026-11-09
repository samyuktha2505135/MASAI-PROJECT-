"""Part C: privacy-first offline support-ticket triage, cost and retry checks."""
import json,re,random,time
from pathlib import Path
import pandas as pd
import mock_llm
R=Path(__file__).resolve().parent; OUT=R/'outputs';REP=R/'reports'; OUT.mkdir(exist_ok=True)
PATS=[('email',r'[\w.+-]+@[\w-]+\.[\w.]+','[EMAIL]'),('phone',r'\b[6-9]\d{9}\b','[PHONE]'),('order_id',r'ORD-\d{6}','[ORDER_ID]')]
ALLOWED={'billing','delivery','technical','fraud'}
ZERO='CLASSIFY support ticket into billing, delivery, technical, fraud. Reply JSON only: {"category": "...", "reason": "..."}.'
FEW=ZERO+'\nEXAMPLES:\nCustomer reports account hacked and fraud -> {"category":"fraud","reason":"unauthorized use"}\nCustomer says app crashes -> {"category":"technical","reason":"app issue"}'
tickets=pd.read_csv(R/'data/raw/tickets.csv',keep_default_na=False)
segments=pd.read_csv(OUT/'customer_segments.csv')
lookup=segments.set_index('customer_id').to_dict('index')
counts={n:0 for n,_,_ in PATS}; redacted={}
for row in tickets.itertuples():
    msg=row.text
    for n,p,rep in PATS:
        msg,nsub=re.subn(p,rep,msg);counts[n]+=nsub
    assert all(re.search(p,msg) is None for _,p,_ in PATS)
    redacted[row.ticket_id]=msg

def decode(txt):
    try:
        d=json.loads(txt)
        if isinstance(d,dict) and d.get('category') in ALLOWED:return d['category'],'classified_json'
        return '','flagged_invalid_output'
    except (ValueError,TypeError):
        words=set(re.findall(r'[a-z]+',(txt or '').lower()))
        matches=words&ALLOWED
        if len(matches)==1:return matches.pop(),'classified_keyword'
        return '','flagged_invalid_output'

def run(prompt,budget=None):
    mock_llm.reset();used=0;total_errors=0;data=[];correct=0;classified=0;conf=[];rng=random.Random(42)
    for row in tickets.itertuples():
        raw=row.text;cat='';tk=0
        if not raw.strip():status='skipped_empty'
        elif mock_llm.est_tokens(raw)>400:status='flagged_long'
        elif budget is not None and used>=budget:status='budget_exhausted'
        else:
            msg=redacted[row.ticket_id]
            assert all(re.search(p,msg) is None for _,p,_ in PATS)
            for attempt in range(4):
                try:
                    response=mock_llm.chat([{'role':'system','content':prompt},{'role':'user','content':msg}]);break
                except mock_llm.TransientLLMError:
                    total_errors+=1
                    if attempt==3: raise
                    delay=.05*(2**attempt)*rng.uniform(.9,1.15)
                    print(f'{row.ticket_id} retry {attempt} wait {delay:.5f}s')
                    time.sleep(delay)
            tk=response.usage.total_tokens;used+=tk
            cat,status=decode(response.choices[0].message.content)
            if cat:
                classified+=1;correct+=cat==row.true_category
                conf.append({'actual':row.true_category,'predicted':cat})
        segment=lookup.get(row.customer_id,{})
        segment_name=segment.get('segment_name','')
        premium=str(segment.get('is_premium','False')).lower()=='true'
        if status not in ('classified_json','classified_keyword'):route,priority='human_review','NORMAL';cat='';segment_name=''
        elif cat=='fraud':route,priority='security_team','HIGH'
        elif premium:route,priority='premium_desk','HIGH'
        else:route,priority='support_queue','NORMAL'
        data.append({'ticket_id':row.ticket_id,'customer_id':row.customer_id,'status':status,'category':cat,'segment_name':segment_name,'priority':priority,'route':route,'tokens_used':tk})
    return pd.DataFrame(data),{'classified':classified,'correct':int(correct),'accuracy':round(correct/classified,4),'tokens':used,'tokens_per_classified':round(used/classified,3),'transient_errors_recovered':total_errors},pd.DataFrame(conf)

zero,zm,conf=run(ZERO);few,fm,_=run(FEW)
budget,bm,_=run(ZERO,6000)
if not (budget.status=='budget_exhausted').any():budget,bm,_=run(ZERO,5000);bm['budget']=5000
else:bm['budget']=6000
final,finalm,_=run(FEW,100000)
final.to_csv(OUT/'triage_results.csv',index=False);budget.to_csv(OUT/'triage_budget_demo.csv',index=False)
confusion=pd.crosstab(conf.actual,conf.predicted);confusion.to_csv(REP/'triage_zero_shot_confusion.csv')
summary={'redactions':{**counts,'total':sum(counts.values())},'zero_shot':zm,'few_shot':fm,'budget_demo':{**bm,'exhausted':int((budget.status=='budget_exhausted').sum())},'final':finalm,'final_status_counts':final.status.value_counts().to_dict(),'final_route_counts':final.route.value_counts().to_dict()}
(REP/'part_c_results.json').write_text(json.dumps(summary,indent=2))
assert len(final)==164 and sum(counts.values())==278
print('PART C DONE:',json.dumps(summary))

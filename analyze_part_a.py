"""Run all seven assessment queries and EDA checks, saving machine-readable findings."""
from pathlib import Path
import json, sqlite3
import pandas as pd
ROOT=Path(__file__).resolve().parent
queries=(ROOT/'sql/queries.sql').read_text().split(';')
queries=[q.strip() for q in queries if q.strip()]
assert len(queries)==7,len(queries)
with sqlite3.connect(ROOT/'data/instamart.db') as db:
    results=[pd.read_sql_query(q,db) for q in queries]
    orders=pd.read_sql_query("SELECT * FROM orders WHERE status='completed'",db)
    customers=pd.read_sql_query('SELECT * FROM customers',db)
    products=pd.read_sql_query('SELECT * FROM products',db)
q1,q2,q3,q4,q5,q6,q7=results
pivot=(orders.merge(customers[['customer_id','city']],on='customer_id')
       .merge(products[['product_id','category']],on='product_id')
       .pivot_table(index='city',columns='category',values='order_amount',aggfunc='sum',margins=True))
city=(orders.merge(customers[['customer_id','city']],on='customer_id')
      .groupby('city').agg(total=('order_amount','sum'),mean=('order_amount','mean'),count=('order_amount','count')))
a=orders.order_amount
q1_count=len(q1)
rank_differences=int((q6.rank_number!=q6.row_number).sum())
q6_top=(q6[q6.rank_number==1][['city','customer_id','total_spend']].to_dict('records'))
corr=orders[['order_amount','discount_pct','quantity','delivery_minutes']].corr()
pairs=[(abs(corr.loc[x,y]),x,y,float(corr.loc[x,y])) for i,x in enumerate(corr.columns) for y in corr.columns[i+1:]]
largest=max(pairs)
q1_,q3_=a.quantile([.25,.75]); iqr=q3_-q1_; lower=q1_-1.5*iqr; upper=q3_+1.5*iqr
report={
 'sql':{'Q1_count':q1_count,'Q2_rows':len(q2),'Q2_total_inr':round(float(q2.revenue.sum()),2),'Q2_largest':q2.iloc[0].to_dict(),
 'Q3_count':len(q3),'Q4_count':len(q4),'Q5_count':len(q5),'Q5_average_inr':round(float(q5.avg_spend.iloc[0]),2),
 'Q6_top_by_city':q6_top,'Q6_rank_row_number_differences':rank_differences,
 'Q7_days':len(q7),'Q7_last_cumulative_inr':round(float(q7.cumulative_revenue.iloc[-1]),2)},
 'pandas':{'pivot_grand_total_inr':round(float(pivot.loc['All','All']),2),'cities':city.round(4).to_dict('index')},
 'eda':{'mean':round(float(a.mean()),4),'median':round(float(a.median()),4),'skewness':round(float(a.skew()),4),
 'q1':round(float(q1_),4),'q3':round(float(q3_),4),'lower_fence':round(float(lower),4),
 'upper_fence':round(float(upper),4),'outliers_below':int((a<lower).sum()),'outliers_above':int((a>upper).sum()),
 'correlations':corr.round(5).to_dict(),'largest_absolute_correlation':{'first':largest[1],'second':largest[2],'r':largest[3]}}}
assert q1_count==30
assert len(q2)==30 and abs(q2.revenue.sum()-2998668.80)<.01
assert len(q3)==11 and len(q4)==25 and len(q5)==550
assert len(q7)==183 and abs(q7.cumulative_revenue.iloc[-1]-2998668.80)<.01
assert rank_differences==8
assert abs(pivot.loc['All','All']-q2.revenue.sum())<.01
(ROOT/'reports/part_a_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('SQL Q1-Q7 passed; analytics saved to reports/part_a_analysis.json')

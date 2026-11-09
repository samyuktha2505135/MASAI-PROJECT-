"""Part A: audit, clean, store and evaluate the original supplied synthetic data."""
from pathlib import Path
import json
import sqlite3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW, CLEAN, REPORTS = ROOT/'data/raw', ROOT/'data/clean', ROOT/'reports'
DB = ROOT/'data/instamart.db'

def audit_frame(df):
    return {'shape': list(df.shape), 'dtypes': {c:str(df[c].dtype) for c in df},
            'missing': {c:{'count':int(df[c].isna().sum()),'percentage':round(float(df[c].isna().mean()*100),3)} for c in df}}

def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    customers = pd.read_csv(RAW/'customers.csv')
    products = pd.read_csv(RAW/'products.csv')
    orders = pd.read_csv(RAW/'orders.csv')
    tickets = pd.read_csv(RAW/'tickets.csv', keep_default_na=False)
    original_orders = orders.copy()
    raw_numbers = {
        'orders_rows':len(orders),'exact_duplicates':int(orders.duplicated().sum()),
        'pending_rows':int(orders.status.eq('pending').sum()),'distinct_order_ids':int(orders.order_id.nunique()),
        'non_numeric_amount':int(pd.to_numeric(orders.order_amount, errors='coerce').isna().sum()),
        'negative_delivery':int((pd.to_numeric(orders.delivery_minutes, errors='coerce')<0).sum()),
        'blank_delivery':int(orders.delivery_minutes.isna().sum()),
        'blank_ages':int(customers.age.isna().sum()),
        'invalid_ages':int((customers.age.notna() & ~customers.age.between(10,100)).sum()),
        'raw_city_spellings':int(customers.city.nunique()),
        'raw_gender_spellings':int(customers.gender.nunique()),
        'blank_genders':int(customers.gender.isna().sum())}
    audit = {'tables': {'customers':audit_frame(customers), 'products':audit_frame(products),
                        'orders':audit_frame(orders),'tickets':audit_frame(tickets)},
             'reference_audit':raw_numbers}
    (REPORTS/'part_a_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')

    customers['city'] = customers.city.str.strip().str.title()
    g = customers.gender.fillna('').astype(str).str.strip().str.lower()
    customers['gender'] = g.map({'m':'male','male':'male','f':'female','female':'female'}).fillna('unknown')
    customers['age'] = pd.to_numeric(customers.age,errors='coerce')
    customers.loc[~customers.age.between(10,100),'age'] = np.nan
    customers['age'] = customers.age.fillna(customers.groupby('city').age.transform('median'))
    customers['signup_date'] = pd.to_datetime(customers.signup_date)

    orders['order_ts'] = pd.to_datetime(orders.order_ts)
    orders = orders.drop_duplicates()
    after_exact = len(orders)
    orders = orders.sort_values(['order_ts'],kind='stable').drop_duplicates('order_id',keep='last').copy()
    after_ids = len(orders)
    orders['order_amount'] = pd.to_numeric(orders.order_amount, errors='coerce')
    amounts_to_fill = int(orders.order_amount.isna().sum())
    prices = products.set_index('product_id').price_inr
    replacement = (orders.product_id.map(prices) * orders.quantity * (1-orders.discount_pct/100)).round(2)
    orders['order_amount'] = orders.order_amount.fillna(replacement)
    orders['delivery_minutes'] = pd.to_numeric(orders.delivery_minutes,errors='coerce')
    negatives_after_dedupe = int(orders.delivery_minutes.lt(0).sum())
    orders.loc[orders.delivery_minutes.lt(0), 'delivery_minutes'] = np.nan
    orders = orders.merge(products[['product_id','category']],on='product_id',how='left',validate='many_to_one')
    done = orders.status.eq('completed')
    median_by_category = orders.loc[done].groupby('category').delivery_minutes.transform('median')
    blanks_to_fill = int(orders.loc[done,'delivery_minutes'].isna().sum())
    orders.loc[done,'delivery_minutes'] = orders.loc[done,'delivery_minutes'].fillna(median_by_category)
    orders.loc[~done,'delivery_minutes'] = np.nan
    orders = orders.drop(columns=['category']).sort_values('order_id').reset_index(drop=True)
    orders['order_ts'] = orders.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')
    customers['signup_date'] = customers.signup_date.dt.strftime('%Y-%m-%d')
    for name,frame in [('customers',customers),('products',products),('orders',orders)]:
        frame.to_csv(CLEAN/f'{name}.csv',index=False)
    with sqlite3.connect(DB) as conn:
        for name,frame in [('customers',customers),('products',products),('orders',orders)]:
            frame.to_sql(name,conn,if_exists='replace',index=False)
    done = orders.status.eq('completed')
    cleanup = {
        'removed_exact_duplicates': len(original_orders)-after_exact,
        'removed_older_order_versions':after_exact-after_ids,
        'remaining_orders':len(orders),'reconstructed_order_amounts':amounts_to_fill,
        'negative_delivery_values_removed_after_deduplication':negatives_after_dedupe,
        'completed_delivery_values_imputed':blanks_to_fill,
        'completed_count':int(done.sum()),'cancelled_count':int((~done).sum()),
        'completed_revenue_inr':round(float(orders.loc[done,'order_amount'].sum()),2),
        'city_counts':customers.city.value_counts().to_dict(),
        'gender_counts':customers.gender.value_counts().to_dict(),
        'customers_age_imputed':int(customers.age.notna().sum())}
    (REPORTS/'part_a_cleaning.json').write_text(json.dumps(cleanup,indent=2),encoding='utf-8')
    print('Part A cleaned',len(customers),'customers and',len(orders),'orders')
    print('Completed revenue INR',cleanup['completed_revenue_inr'])
    return customers,products,orders

if __name__=='__main__': main()

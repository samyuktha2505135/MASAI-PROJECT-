"""Part B: leakage-aware RFM segmentation and customer lapse forecasting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from threshold_model import ThresholdedModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
REPORT=ROOT/'reports'; REPORT.mkdir(exist_ok=True)
orders=pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts'])
customers=pd.read_csv(ROOT/'data/clean/customers.csv',parse_dates=['signup_date'])
products=pd.read_csv(ROOT/'data/clean/products.csv')
orders=orders.merge(products[['product_id','category']],on='product_id',how='left',validate='many_to_one')
completed=orders.loc[orders.status.eq('completed')].copy()
snapshot=pd.Timestamp('2025-10-01')
rfm=completed.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
rfm['recency']=(snapshot-rfm.last_order).dt.days
rfm=rfm[['customer_id','recency','frequency','monetary']]
scaler=StandardScaler(); scaled=scaler.fit_transform(rfm[['recency','frequency','monetary']])
ks=[]
for k in range(2,9):
    km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(scaled)
    ks.append({'k':k,'inertia':round(float(km.inertia_),3),'silhouette':round(float(silhouette_score(scaled,km.labels_)),4)})
pd.DataFrame(ks).to_csv(REPORT/'kmeans_scores.csv',index=False)
final_k=4
km=KMeans(n_clusters=final_k,n_init=10,random_state=42).fit(scaled)
rfm['segment_id']=km.labels_
profile=rfm.groupby('segment_id').agg(size=('customer_id','count'),recency=('recency','mean'),frequency=('frequency','mean'),monetary=('monetary','mean')).reset_index()
premium_id=int(profile.loc[profile.monetary.idxmax(),'segment_id'])
# Labels are data-derived business descriptions; cluster IDs are arbitrary.
name_map={}
for _,r in profile.iterrows():
    idx=int(r.segment_id)
    if idx==premium_id: name='Premium'
    elif r.recency>=profile.recency.median(): name='At-Risk / Infrequent'
    elif r.frequency>=profile.frequency.median(): name='Frequent Shoppers'
    else: name='Regular Shoppers'
    name_map[idx]=name
rfm['segment_name']=rfm.segment_id.map(name_map)
rfm['is_premium']=rfm.segment_id.eq(premium_id)
rfm.to_csv(OUT/'customer_segments.csv',index=False)
profile['segment_name']=profile.segment_id.map(name_map)
profile.to_csv(REPORT/'segment_profile.csv',index=False)
pca=PCA(n_components=2).fit(scaled)
# Pre-cutoff cohort and features; September orders used ONLY to assign the target.
cutoff=pd.Timestamp('2025-09-01'); end=pd.Timestamp('2025-10-01')
pre=orders.loc[orders.order_ts.lt(cutoff)].copy()
pre_c=pre.loc[pre.status.eq('completed')].copy()
cohort=pre_c.customer_id.drop_duplicates().sort_values().tolist()
feat=pre_c.groupby('customer_id').agg(last=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount_pct=('discount_pct','mean'),avg_delivery_minutes=('delivery_minutes','mean'))
feat['recency']=(cutoff-feat['last']).dt.days
feat=feat.drop(columns=['last'])
cancel=pre.assign(cancelled=pre.status.eq('cancelled').astype(float)).groupby('customer_id').cancelled.mean()
feat['cancellation_rate']=cancel
fav=pre_c.groupby(['customer_id','category']).size().reset_index(name='count').sort_values(['customer_id','count','category'],ascending=[True,False,True]).drop_duplicates('customer_id').set_index('customer_id')['category']
feat['favorite_category']=fav
feat=feat.join(customers.set_index('customer_id')[['city','age','gender','signup_date']],how='left')
feat['tenure_days']=(cutoff-feat.signup_date).dt.days
feat=feat.drop(columns=['signup_date'])
sept=set(completed.loc[completed.order_ts.ge(cutoff)&completed.order_ts.lt(end),'customer_id'])
# Additional pre-cutoff behaviour: never use September orders as model inputs.
for days in [14, 30, 60, 90]:
    window = pre_c.loc[pre_c.order_ts.ge(cutoff - pd.Timedelta(days=days))]
    recent = window.groupby('customer_id').agg(**{
        f'frequency_{days}d': ('order_id', 'nunique'),
        f'monetary_{days}d': ('order_amount', 'sum')})
    feat = feat.join(recent)
    feat[[f'frequency_{days}d', f'monetary_{days}d']] = feat[[f'frequency_{days}d', f'monetary_{days}d']].fillna(0)
feat['orders_per_30d'] = feat.frequency / (feat.tenure_days.clip(lower=1) / 30)
feat['average_order_value'] = feat.monetary / feat.frequency
feat['activity_gap_ratio'] = feat.recency / feat.tenure_days.clip(lower=1)
feat['last_order_weekday'] = pre_c.sort_values('order_ts').drop_duplicates('customer_id', keep='last').set_index('customer_id').order_ts.dt.dayofweek
feat['avg_quantity'] = pre_c.groupby('customer_id').quantity.mean()
feat['max_order_value'] = pre_c.groupby('customer_id').order_amount.max()
feat['order_std_value'] = pre_c.groupby('customer_id').order_amount.std().fillna(0)
feat['unique_product_count'] = pre_c.groupby('customer_id').product_id.nunique()
feat['active_days'] = pre_c.groupby('customer_id').order_ts.agg(lambda series: series.dt.date.nunique())
feat['lapsed']=(~feat.index.isin(sept)).astype(int)
assert set(feat.index)==set(cohort)
assert not feat.isna().any().any(),feat.isna().sum()
X=feat.drop(columns='lapsed'); y=feat.lapsed
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
nums=[column for column in X.columns if column not in ['favorite_category', 'city', 'gender']]
cats=['favorite_category','city','gender']
def make_pipe(model):
    return Pipeline([('preprocess',ColumnTransformer([('nums',StandardScaler(),nums),('cats',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False),cats)])),('model',model)])
log_grid=GridSearchCV(make_pipe(LogisticRegression(max_iter=1000)),{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=1)
rf_grid=GridSearchCV(make_pipe(RandomForestClassifier(n_estimators=100,random_state=42)),{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=1)
log_grid.fit(Xtr,ytr);rf_grid.fit(Xtr,ytr)
def metrics(model):
    pred=model.predict(Xte)
    return {'precision':round(float(precision_score(yte,pred,zero_division=0)),4),'recall':round(float(recall_score(yte,pred,zero_division=0)),4),'f1':round(float(f1_score(yte,pred,zero_division=0)),4),'confusion_matrix':confusion_matrix(yte,pred).tolist()}
# Choose a decision threshold from TRAINING predictions alone. The held-out test
# is not used for thresholds, model hyperparameters, scaling or feature creation.
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
threshold_report={}
def tune_threshold(estimator,name):
    train_prob=cross_val_predict(estimator,Xtr,ytr,cv=cv,method='predict_proba',n_jobs=1)[:,1]
    thresholds=np.arange(0.25,0.701,0.025)
    candidate=[(float(f1_score(ytr,train_prob>=threshold)),float(threshold)) for threshold in thresholds]
    best_f1,best_threshold=max(candidate,key=lambda pair:(pair[0],-abs(pair[1]-.5)))
    threshold_report[name]={'training_oof_f1':round(best_f1,4),'threshold':round(best_threshold,4)}
    return ThresholdedModel(estimator,best_threshold)
log_adjusted=tune_threshold(log_grid.best_estimator_,'logistic')
rf_adjusted=tune_threshold(rf_grid.best_estimator_,'forest')
log_m=metrics(log_adjusted);rf_m=metrics(rf_adjusted)
# Balance only training examples, not test examples or the feature transform.
rng=np.random.RandomState(42)
minority=ytr[ytr.eq(1)].index
n_more=int((ytr==0).sum()-(ytr==1).sum())
extra=rng.choice(minority,size=max(0,n_more),replace=True)
Xbal=pd.concat([Xtr,Xtr.loc[extra]]); ybal=pd.concat([ytr,ytr.loc[extra]])
balanced=make_pipe(LogisticRegression(max_iter=1000,C=log_grid.best_params_['model__C']))
balanced.fit(Xbal,ybal)
# Pick the deployed model using training-only out-of-fold F1, never test F1.
selected=rf_adjusted if threshold_report['forest']['training_oof_f1']>threshold_report['logistic']['training_oof_f1'] else log_adjusted
selected_name='RandomForest' if selected is rf_adjusted else 'LogisticRegression'
joblib.dump(selected,OUT/'lapse_model.joblib')
assert np.array_equal(joblib.load(OUT/'lapse_model.joblib').predict(Xte),selected.predict(Xte))
test_ids=Xte.index.to_list()
result=pd.DataFrame({'customer_id':test_ids,'lapse_probability':selected.predict_proba(Xte)[:,1],'actually_lapsed':yte.to_numpy()})
result=result.merge(rfm[['customer_id','is_premium']],on='customer_id',how='left')
retention=result[result.is_premium.eq(True)].sort_values('lapse_probability',ascending=False).head(10).drop(columns='is_premium')
retention.to_csv(OUT/'retention_list.csv',index=False)
feature_names=rf_grid.best_estimator_.named_steps['preprocess'].get_feature_names_out()
importances=rf_grid.best_estimator_.named_steps['model'].feature_importances_
top5=[{'feature':str(feature_names[i]),'importance':round(float(importances[i]),5)} for i in np.argsort(importances)[::-1][:5]]
coefs=log_grid.best_estimator_.named_steps['model'].coef_[0]
features_log=log_grid.best_estimator_.named_steps['preprocess'].get_feature_names_out()
log_signs={str(features_log[i]):('positive' if coefs[i]>0 else 'negative') for i in range(min(3,len(coefs)))}
train_mean=float(Xtr.frequency.mean())
actual={'rfm_customers':len(rfm),'rfm_recency_median':float(rfm.recency.median()),'rfm_frequency_mean':round(float(rfm.frequency.mean()),4),'rfm_monetary_mean':round(float(rfm.monetary.mean()),2),'kmeans':ks,'chosen_k':final_k,'premium_segment_id':premium_id,'pca_explained_variance':pca.explained_variance_ratio_.round(5).tolist(),'cohort':len(feat),'lapsed':int(y.sum()),'train':len(Xtr),'test':len(Xte),'baseline_accuracy':round(float(accuracy_score(yte,np.zeros(len(yte)))),4),'baseline_f1':0,'frequency_train_mean':train_mean,'frequency_full_cohort_mean':float(X.frequency.mean()),'logistic':{'best_params':log_grid.best_params_,'best_cv_f1':round(float(log_grid.best_score_),4),'test':log_m},'forest':{'best_params':rf_grid.best_params_,'best_cv_f1':round(float(rf_grid.best_score_),4),'test':rf_m,'top5_features':top5},'balanced_logistic_test':metrics(balanced),'logistic_coefficient_signs':log_signs,'selected':selected_name,'threshold_tuning':threshold_report,'selected_threshold':round(float(selected.threshold),4),'retention_count':len(retention),'retention_actually_lapsed':int(retention.actually_lapsed.sum())}
(REPORT/'part_b_results.json').write_text(json.dumps(actual,indent=2))
print('PART B DONE: cohort',len(feat),'lapsed',int(y.sum()),'selected',selected_name,'test F1',metrics(selected)['f1'])

# Instamart Customer Intelligence & Support Copilot

## Overview

A four-part, **offline and fully synthetic**, reproducible academic capstone: Part A audits/cleans order exports and constructs an SQLite analytical database; Part B builds RFM customer segments and a held-out lapse prediction model; Part C triages tickets using a privacy-first offline mock LLM; Part D implements TF-IDF policy retrieval and a bounded, resumable simulated refund agent. **No real Swiggy Instamart customer records, policy or money transfers are involved.** Values are in INR.

The supplied `generate_data.py` and `mock_llm.py` are included unmodified. Only the Python standard library, pandas, numpy, scikit-learn and joblib are used. Python 3.11+ and SQLite 3.25+ are required. `notebooks/Customer_Intelligence_Support_Copilot.ipynb` is optional exploration; `.py` modules are the reproducible pipeline. Reports under `reports/` contain machine-readable evidence for all numbers below.

## Reproduce

From a freshly cloned project repository, use the **numbered steps** below (commands shown for Windows PowerShell; Python 3.11+ is required). A public GitHub repository with the final files on `main` must be created and pushed separately.

1. `python -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1` (macOS/Linux: `source .venv/bin/activate`)
3. `python -m pip install -r requirements.txt`
4. `python generate_data.py`
5. `python part_a.py`
6. `python validate_part_a.py`
7. `python analyze_part_a.py`
8. `python part_b.py`
9. `python part_c.py`
10. `python part_d.py`
11. `python integration_tests.py`

Alternatively, steps 4–11 are automated by `python run_project.py`. The final command prints `ALL INTEGRATION CHECKS PASSED`, and the runner prints `ALL PROJECT STAGES COMPLETED`. `generate_data.py` prints `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. `validate_part_a.py` prints exactly `ALL CHECKPOINTS PASSED`. The generated CSV files, SQLite database, trained joblib model, JSONL audit logs and checkpoint state are included in this repository.

The pipeline can be rerun without a network connection after dependencies are installed. Exact package versions are pinned in `requirements.txt`. Input generation is seeded, both clustering and prediction are seeded, and ticket-model behavior is deterministic.

## Part A

### Raw audit

All tables, data types, and per-column missing-value counts and percentages are recorded in `reports/part_a_audit.json`.

| Audit item | Value |
|---|---:|
| Raw customers/products/orders/tickets | 1,200 / 40 / 13,084 / 164 |
| Raw order exact duplicates | 119 |
| Pending order records | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery values | 106 |
| Blank delivery values | 2,541 |
| Blank / out-of-range ages | 61 / 6 |
| Raw city spellings | 24 |
| Nonblank raw gender spellings | 10 |
| Blank genders | 30 |

### Cleaning rules and changes

1. Strip city whitespace and title-case spellings: 24 raw spellings become six cities. Result counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184.
2. Map gender spellings to `male`, `female`, `unknown` (blank values become `unknown`): female 610, male 560, unknown 30.
3. Mark ages outside 10–100 as missing (six ages); impute the original 61 blanks plus six invalid values using each city's median (`groupby('city').transform('median')`). No ages remain missing; all final ages are between 18 and 60.
4. Parse `signup_date` and `order_ts` as datetime. Save dates in ISO format and timestamps in `YYYY-MM-DD HH:MM:SS` format.
5. Remove 119 exact duplicate order rows, then discard 656 older order versions by keeping the row with the latest `order_ts` for each `order_id`.
6. Convert `order_amount` to numeric; fill 112 values that remain missing *after deduplication* with `round(price_inr * quantity * (1 - discount_pct/100), 2)`.
7. Remove 105 negative delivery values remaining *after deduplication*; impute 774 missing completed-order delivery values with the median delivery time of the product category among completed orders. Preserve all 1,199 cancelled-order delivery times as missing.
8. Export clean CSV files to `data/clean/`, and three SQLite tables (`customers`, `products`, `orders`) to `data/instamart.db`.

Final dataset: **12,309** orders, comprising **11,110 completed** and **1,199 cancelled**; zero pending. Completed-order revenue: **INR 2,998,668.80**. No missing completed-order delivery times or order amounts. Detailed change counts: `reports/part_a_cleaning.json`.

### Seven SQL queries

The following results were executed via `pd.read_sql_query` and saved in `reports/part_a_analysis.json`.

| Query | Actual verified result |
|---|---|
| Q1 WHERE + HAVING | 30 customers |
| Q2 Three-table join | 30 city-category cells; INR 2,998,668.80 total; largest Bengaluru / Electronics INR 233,909.45 |
| Q3 Scalar subquery | 11 products |
| Q4 LEFT JOIN | 25 customers with no completed orders |
| Q5 CTE | 550 customers above mean completed spend of INR 2,552.06 |
| Q6 Window ranks | Highest-spend customer by city: Bengaluru C0082 INR 12,856.20; Chennai C0930 INR 10,151.40; Delhi C0291 INR 13,742.60; Hyderabad C0743 INR 12,730.05; Mumbai C0693 INR 14,766.55; Pune C0426 INR 10,377.10. Eight rows have different RANK and ROW_NUMBER values. |
| Q7 Running total | 183 days; ending cumulative revenue INR 2,998,668.80 |

`RANK()` allows tied spend values to share a rank and skips subsequent ranking numbers; `ROW_NUMBER()` assigns a unique sequence to every row even when spend ties.

### Pandas and EDA cross-check

Pandas `pivot_table(..., margins=True)` returns the same INR 2,998,668.80 grand total as Q2. City-level named aggregates include Bengaluru INR 523,643.40 total, mean INR 276.47, 1,894 orders; Mumbai INR 479,058.50 total, mean INR 297.00, 1,613 orders. All city aggregates are in the analysis JSON.

For completed orders: mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, Q3 INR 297.00, and 1.5×IQR lower/upper fences INR -283.50 / INR 645.30. There are no observations below the lower fence and 897 above the upper fence.

**Four EDA questions (under 200 words):** The data represents synthetic Instamart customers, product purchases and order fulfilment. Revenue is strongly right-skewed, with the mean INR 269.91 substantially above the median INR 135.00, consistent with a smaller number of high-value electronics purchases. There are 897 upper-fence price outliers, while planted impossible delivery durations, missing amounts and duplicate order rows were corrected using the prescribed rules. The strongest absolute pairwise correlation among order amount, discount, quantity and delivery time is order amount versus discount percentage (r approximately -0.095), only slightly larger in magnitude than order amount versus quantity (r approximately +0.087). **Correlations this small do not establish causation.**


## Part B

### RFM and segmentation

Using **completed orders only** and the 2025-10-01 snapshot, the RFM table has **1175** customers, recency median **19** days, mean frequency **9.46** and mean spend **INR 2,552.06**. Recency is whole days since last completed order; frequency is the count of distinct completed orders; monetary is summed completed spend.

KMeans uses StandardScaler, `random_state=42` and `n_init=10` on the RFM columns:

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.76 | 0.4650 |
| 3 | 1268.23 | 0.4452 |
| 4 | 743.07 | 0.5255 |
| 5 | 617.37 | 0.4805 |
| 6 | 503.17 | 0.4819 |
| 7 | 408.46 | 0.4163 |
| 8 | 366.50 | 0.4123 |

We choose **K=4**: the pronounced inertia reduction at K=4 combines with the strongest silhouette in the table, and four customer groups permit distinct business campaigns. K=8 has lower inertia because additional centroids cannot increase the optimized within-cluster sum of squared distances; that alone does not prove its segments are more meaningful or actionable. Cluster IDs are arbitrary labels, not rankings.

| Segment ID | Business name | Customers | Mean recency | Mean frequency | Mean spend (INR) |
|---:|---|---:|---:|---:|---:|
| 0 | At-Risk / Infrequent | 436 | 28.86 | 3.77 | 1057.62 |
| 1 | Premium | 138 | 24.12 | 6.77 | 7129.70 |
| 2 | Frequent Shoppers | 419 | 8.77 | 19.40 | 3410.02 |
| 3 | At-Risk / Infrequent | 182 | 109.04 | 2.20 | 685.99 |

Business actions: contact **At-Risk / Infrequent** customers with win-back offers; reward **Frequent Shoppers** with loyalty perks; maintain **Regular Shoppers** with cross-selling; offer the highest-monetary **Premium** segment dedicated support and retention calls. Only the highest mean monetary cluster is labelled Premium. Two-component PCA explained variance: **0.634** and **0.218**, totaling **0.852**. Output: `outputs/customer_segments.csv`.

### Lapse label, training and validation

Cutoff: **2025-09-01 00:00:00**. Cohort is customers with completed orders *before* that cutoff. The `lapsed=1` label means no completed order during September 2025. Feature data consists **only of pre-cutoff orders**, with recency, frequency, monetary spend, average discount, cancellation rate, average completed delivery time, most common product category, city, age, gender and tenure. Additional pre-cutoff features include 14/30/60/90-day order frequency and monetary totals, purchasing rate, average order value, inactivity-to-tenure ratio, most recent weekday, average quantity, maximum and standard deviation of order value, unique product count and active ordering days. **No order on or after 2025-09-01 entered the model features; neither whole-window segment labels nor September activity entered those features.** Missing feature values are asserted absent.

The cohort has **1156** customers, **443** lapsed (38.32%), **924** training and **232** held-out test rows. Stratified 80/20 split uses seed 42. Training-only one-hot encoding (`drop='first'`) and scaling are encapsulated in a scikit-learn Pipeline. **No scaler, encoder or resampler saw held-out test rows.** Learned training frequency mean: **8.11472**; full-cohort mean: **8.02855**; they differ because the training subset is a random sample, not the entire cohort. Always predicting not lapsed yields accuracy **61.64%** and positive-class F1 **0.00**.

LogisticRegression (`max_iter=1000`) uses 5-fold GridSearchCV over `C=[0.01,0.1,1,10,100]`. RandomForest (`n_estimators=100,random_state=42`) uses 5-fold GridSearchCV over max depth 3/5/8 and min leaf samples 5/10/20 (45 fits). Both optimize positive-class F1 and are tuned before the untouched test is evaluated.

| Model | Best parameters | Grid CV F1 (threshold 0.5) | Training OOF-selected threshold | Held-out precision | Held-out recall | Held-out F1 | Confusion matrix [[TN,FP],[FN,TP]] |
|---|---|---:|---:|---:|---:|---:|---|
| Logistic | C=0.1 | 0.6499 | 0.30 | 0.5608 | 0.9326 | 0.7004 | [[78, 65], [6, 83]] |
| Forest | depth=3, leaf=5 | 0.6897 | 0.30 | 0.5556 | 0.9551 | **0.7025** | [[75, 68], [4, 85]] |

**Training-only threshold tuning:** After completing the original specified hyperparameter grids, we generated 5-fold out-of-fold probabilities from the 924 training examples (`StratifiedKFold`, shuffle=True, random_state=42). The threshold grid 0.25 to 0.70 in increments of 0.025 maximized training OOF F1, giving 0.30 for each model. Training-only OOF F1 was 0.7186 for Logistic Regression and 0.7190 for Random Forest. The Random Forest was selected **on training OOF F1 alone**, not from held-out test comparisons. After model selection and all threshold tuning, the untouched test was evaluated once for reporting. The test result, F1=0.7025, is from the same fixed 232-row stratified held-out set. The lower threshold increases sensitivity to lapsing customers but increases false-positive retention offers. For the selected forest, held-out *accuracy* is (75+85)/232 = **68.97%**; do not confuse this with 70.25% F1. The original version had 64.80% F1 and 72.84% accuracy. Thus the improvement is in **F1 and recall, not accuracy**.

The fitted model and decision threshold are saved together in `outputs/lapse_model.joblib` using `ThresholdedModel` from `threshold_model.py`; reload with that module available. We verify exact equality of predictions after `joblib.load`. Train-only over-sampling, separate from threshold tuning, to equal class sizes changes Logistic Regression held-out precision/recall/F1 to **0.5588 / 0.8539 / 0.6756**. The majority of the movement is in recall; false-positive outreach is the cost of reaching more lapsed customers.

Selected **RandomForest** because its training OOF F1 slightly exceeds the Logistic Regression alternative. Missing a lapsing customer risks losing future purchases and raises reacquisition costs; falsely contacting a loyal customer uses an incentive or staff time. A recall-focused threshold is appropriate for a low-cost retention outreach campaign, but this model should not trigger expensive incentives without considering precision. Neither the chosen threshold nor the reported F1 establishes performance on real customers outside this synthetic dataset.

Forest top features: `nums__frequency` (0.17429), `nums__unique_product_count` (0.16385), `nums__frequency_90d` (0.13037), `nums__active_days` (0.08349), `nums__monetary` (0.08234). These measure historical purchasing intensity and breadth; fitted importances are associations, not causal conclusions. Logistic coefficient signs for frequency (negative), monetary (positive) and average discount (positive) likewise are conditional associations, not evidence of causality.

`outputs/retention_list.csv` contains **10** Premium test customers ranked by predicted lapse probability, **0** of whom actually lapsed under the new model ranking. **The retention list is illustrative, not a leakage-free evaluation:** its Premium flag is a post-hoc whole-window segment using September orders, even though model inputs use only pre-cutoff features.

## Part C

Original 164 synthetic tickets are read with `keep_default_na=False`. Four empty texts are `skipped_empty`; four very long texts (`est_tokens > 400`) are `flagged_long`; neither enters the model. Email, phone and order-ID substitutions are applied in the prescribed order and only redacted text is passed as the model user message. Regex assertions confirm zero personal-detail patterns remaining. Redactions: **55 emails, 91 phones, 132 order IDs = 278** total.

The two JSON-only prompts (zero-shot and few-shot) use the unmodified `mock_llm.py`. Transient failures are retried with exponential backoff and randomized jitter (`0.05 * 2**n * uniform(0.9,1.15)`, up to three retries). Each full eligible-ticket run recovered **19** transient errors. Valid JSON categories are accepted; chatty answers are only accepted when `re.findall(r'[a-z]+', ...)` contains exactly one allowed category. An invalid JSON category is never guessed: it goes to human review. Across 164 tickets the final status counts are **135 JSON-classified, 12 keyword-classified, 9 invalid output, 4 long, 4 empty**.

| Prompt | Classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 147 | 126 | 85.71% | 7528 | 51.21 |
| Few-shot | 147 | 147 | 100.00% | 10960 | 74.56 |

The zero-shot confusion table is in `reports/triage_zero_shot_confusion.csv`. The extra few-shot cost is **23.35 tokens per classified ticket**, for 21 additional correct mock classifications. With this synthetic model the premium in tokens is worthwhile if routing mistakes cost more than token usage; on real tickets the trade-off would need independent validation. The final few-shot run uses a 100,000-token budget. The **6000**-token zero-shot demo leaves **31** tickets as `budget_exhausted` and sends them to humans, never calling the model after its running usage passes the budget.

Routing reads Part B premium flags: fraud → HIGH/security_team; otherwise Premium → HIGH/premium_desk; other valid tickets → NORMAL/support_queue; skipped, long, invalid, budget-exhausted → NORMAL/human_review. Final routing: **{'support_queue': 102, 'security_team': 34, 'human_review': 17, 'premium_desk': 11}**. The policy of conservative human review means the 100% synthetic few-shot accuracy applies only to the 147 accepted classifications, not all 164 tickets. Output files: `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv`.

## Part D

Five supplied policies are split into 40-word chunks with 10 words overlapping (step size 30), producing **15 chunks**: refund 4, delivery 3, escalation 3, payment 3, privacy 2. TF (term frequency) measures a word's prominence within a chunk. IDF (inverse document frequency) downweights terms appearing in many chunks so rare terms matter more. TF-IDF cannot reliably connect synonyms with no shared words. `TfidfVectorizer` and `cosine_similarity` retrieve the top three policy chunks; all three chunks for `refund policy late delivery` are from refund_policy.txt.

The POLICY ASSISTANT receives the three source-labelled chunks as `CONTEXT:` and the user question as `QUESTION:`. The 14-question evaluation produces **12/12** expected sources, **12/12** answer facts and **2/2** fallbacks for out-of-policy questions. With **zero overlap**, fact coverage drops to **10/12**: policy facts around chunk boundaries become split across text windows. The per-question failures are in `reports/part_d_policy_evaluation.json`.

The ReAct refund agent uses five explicit tool schemas (`get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`) and `TOOL_MAP`. Only completed orders delivered **more than 45 minutes** late get a simulated full refund; amounts above **INR 1,000** are escalated and not automatically issued. `issue_refund` and escalation append JSON lines to `outputs/refund_ledger.jsonl`, with UTC timestamps. The model is supplied offline by the unchanged mock file, and the agent never transfers real money.

| Scenario | Status | Model calls | Tokens used |
|---|---|---:|---:|
| S1 | REFUND | 5 | 907 |
| S2 | REFUND | 5 | 907 |
| S3 | ESCALATE | 5 | 894 |
| S4 | DENIED | 4 | 613 |
| S5 | DENIED | 3 | 343 |
| S6 | NOT_FOUND | 2 | 76 |
| S7 | FALLBACK_MAX_ITERATIONS | 6 | 3012 |

The expected simulated values for S1/S2/S3 are **INR 128.25**, **INR 135.00** and **INR 1,999.00** respectively; S4/S5 are denied, S6 is not found, and S7 runs to `FALLBACK_MAX_ITERATIONS` after six calls. The full observations and status/results are in `outputs/agent_runs.jsonl`. Safety demonstrations: S1 with token_limit=200 produces **FALLBACK_TOKEN_LIMIT**, and time_limit_seconds=0 produces **FALLBACK_TIMEOUT** before the first model call. All fallback outcomes return a human-handoff summary and preserve completed steps.

A checkpoint is written atomically after each completed tool observation and stores run id, scenario id, message history, tokens, step count and executed tool list in `state/`. For the S1 simulated crash immediately after the second tool, `resume_agent` continues with exactly these tools in order: **get_order, search_policy, calculate_refund, issue_refund**, with no repeat; final status **REFUND** and refund **INR 128.25**.

**ReAct vs plan-and-execute (under 250 words).** S1 uses 5 LLM calls, S5 uses 3 and S6 uses 2, while a plan-and-execute approach could ask for a full plan once and only replan on failure, often reducing calls and tokens. However, an unexpected NOT_FOUND observation or a cancelled order must stop any intended refund before money is issued; the ReAct loop evaluates each new observation instead of assuming an initial plan remains valid. This increases model calls but makes it clearer to enforce action-by-action constraints and checkpoint after effects. For a refund workflow with authorization boundaries, a bounded ReAct agent with explicit deterministic tool validation and human escalation is preferable to blindly executing a preplanned chain.

## Limitations and Next Steps

All customer data, policies, requests and model results are synthetic. Part C's 100% few-shot classification is specific to the provided deterministic offline mock, not a generalization claim. Part B lapse features use only pre-cutoff information, but its **Premium retention list is illustrative and not a leakage-free evaluation**, because the Premium flag is computed from whole-window RFM including September. The policy QA mock is retrieval-grounded, but its rules and token estimates are simplified; the fallback for unrelated questions is conservative. The simulated refund ledger is not a real payment system and should not be used with real customer or financial data. The checkpoint demo demonstrates single-run recovery; production execution would also need concurrency locking, durable transaction IDs, idempotent financial operations and access controls.

The model was evaluated on the assessment's prescribed synthetic held-out split, not live deployment data. Future work would include out-of-time testing, bias checks, stronger identity authorization, operational monitoring and review of ambiguous policy requests. Keep both original supplied files unchanged, commit the required outputs to `main` and submit **only the public GitHub link pointing to `/tree/main`**. The brief does not accept PDFs, screenshots, slide decks, videos, or ZIPs as the graded deliverable.

-- Q1: WHERE and HAVING
SELECT customer_id, SUM(order_amount) AS filtered_spend
FROM orders WHERE status='completed' AND order_amount > 500
GROUP BY customer_id HAVING SUM(order_amount) > 8000
ORDER BY filtered_spend DESC;

-- Q2: three-table join
SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue
FROM orders o JOIN customers c ON o.customer_id=c.customer_id
JOIN products p ON o.product_id=p.product_id
WHERE o.status='completed'
GROUP BY c.city,p.category ORDER BY revenue DESC;

-- Q3: scalar subquery
SELECT product_id,product_name,price_inr FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC;

-- Q4: LEFT JOIN
SELECT c.customer_id FROM customers c
LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed'
WHERE o.order_id IS NULL ORDER BY c.customer_id;

-- Q5: CTE + above-average customer spend
WITH spending AS (
 SELECT customer_id,SUM(order_amount) AS total_spend FROM orders
 WHERE status='completed' GROUP BY customer_id
), average_spending AS (SELECT AVG(total_spend) AS avg_spend FROM spending)
SELECT s.customer_id,s.total_spend,a.avg_spend FROM spending s
CROSS JOIN average_spending a WHERE s.total_spend > a.avg_spend ORDER BY s.total_spend DESC;

-- Q6: ranking by city
WITH spending AS (
 SELECT c.city,o.customer_id,SUM(o.order_amount) AS total_spend
 FROM orders o JOIN customers c ON c.customer_id=o.customer_id
 WHERE o.status='completed' GROUP BY c.city,o.customer_id
)
SELECT city,customer_id,total_spend,
 RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rank_number,
 DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank_number,
 ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_number
FROM spending ORDER BY city,rank_number,customer_id;

-- Q7: cumulative daily revenue
WITH daily AS (
 SELECT SUBSTR(order_ts,1,10) AS day,SUM(order_amount) AS revenue
 FROM orders WHERE status='completed' GROUP BY SUBSTR(order_ts,1,10)
)
SELECT day,revenue,SUM(revenue) OVER (ORDER BY day) AS cumulative_revenue
FROM daily ORDER BY day;

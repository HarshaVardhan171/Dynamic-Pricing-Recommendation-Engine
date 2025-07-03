CREATE DATABASE ecom_pricing;
USE ecom_pricing;
-- Table Setup
SELECT
    COUNT(*) AS total_records,
    COUNT(DISTINCT Product_ID) AS unique_products,
    SUM(Final_Price * Units_Sold) AS total_revenue,
    AVG(Final_Price) AS avg_final_price,
    AVG(Units_Sold) AS avg_units_sold
FROM `dynamic pricing recom engine`;

-- Column Name Changed
SELECT * FROM `dynamic pricing recom engine` LIMIT 10;
SHOW COLUMNS FROM `dynamic pricing recom engine`;
ALTER TABLE `dynamic pricing recom engine` CHANGE `ï»¿Row Number` `Row Number` INT;
DESCRIBE `dynamic pricing recom engine`;

-- Monthly Revenue and Sales Trend
SELECT
    DATE_FORMAT(Date, '%Y-%m') AS month,
    SUM(Final_Price * Units_Sold) AS revenue,
    SUM(Units_Sold) AS total_units
FROM `dynamic pricing recom engine`
GROUP BY month
ORDER BY month;

SELECT
    MONTHNAME(Date) AS month,
    SUM(Final_Price * Units_Sold) AS revenue,
    SUM(Units_Sold) AS total_units
FROM `dynamic pricing recom engine`
GROUP BY month;

-- Promo & Discount Effectiveness

SELECT
    Promotion,
    ROUND(AVG(Units_Sold), 2) AS avg_units_sold,
    ROUND(AVG(Final_Price * Units_Sold), 2) AS avg_revenue
FROM `dynamic pricing recom engine`
GROUP BY Promotion;

SELECT
    CASE
        WHEN Discount_Percent < 10 THEN '0–9%'
        WHEN Discount_Percent < 20 THEN '10–19%'
        WHEN Discount_Percent < 30 THEN '20–29%'
        WHEN Discount_Percent < 40 THEN '30–39%'
        ELSE '40%+'
    END AS discount_range,
    ROUND(AVG(Units_Sold), 1) AS avg_units_sold,
    COUNT(*) AS product_count
FROM `dynamic pricing recom engine`
GROUP BY discount_range
ORDER BY discount_range;

-- Price Sensitivity by Category

SELECT
    Category,
    ROUND(AVG(Final_Price), 2) AS avg_price,
    ROUND(AVG(Units_Sold), 2) AS avg_units_sold
FROM `dynamic pricing recom engine`
GROUP BY Category
ORDER BY avg_units_sold DESC;

SELECT
    Category,
    CASE
        WHEN Final_Price < 500 THEN '0–499'
        WHEN Final_Price < 1000 THEN '500–999'
        WHEN Final_Price < 1500 THEN '1000–1499'
        WHEN Final_Price < 2000 THEN '1500–1999'
        WHEN Final_Price < 2500 THEN '2000–2499'
        ELSE '2500+'
    END AS price_band,
    SUM(Units_Sold) AS total_units
FROM `dynamic pricing recom engine`
GROUP BY Category, price_band
ORDER BY Category, price_band;

-- Competitor Price Impact
SELECT
    CASE
        WHEN Final_Price < Competitor_Price THEN 'We are cheaper'
        WHEN Final_Price > Competitor_Price THEN 'We are expensive'
        ELSE 'Equal Price'
    END AS price_position,
    ROUND(AVG(Units_Sold), 1) AS avg_units_sold
FROM `dynamic pricing recom engine`
GROUP BY price_position;

-- Sweet Spot Discovery

SELECT
    CASE
        WHEN Final_Price < 500 THEN '0–499'
        WHEN Final_Price < 1000 THEN '500–999'
        WHEN Final_Price < 1500 THEN '1000–1499'
        WHEN Final_Price < 2000 THEN '1500–1999'
        WHEN Final_Price < 2500 THEN '2000–2499'
        ELSE '2500+'
    END AS price_band,
    SUM(Units_Sold) AS total_units_sold,
    ROUND(AVG(Final_Price * Units_Sold), 2) AS avg_revenue
FROM `dynamic pricing recom engine`
GROUP BY price_band
ORDER BY price_band;







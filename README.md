# 📈 Job Market Insights & Salary Analysis

## 📊 **Project Overview**
---
This project provides a deep-dive analysis into a modern **Job Market Dataset**. By leveraging Python's data science stack, it explores the relationships between **salaries, geographic locations, required skills, and experience levels**. 

The goal is to provide actionable insights for job seekers and recruiters while preparing the data for predictive machine learning models.

## 🔍 **Key Insights & Interpretations**
---

### 1. **Data Completeness & Structure**
> **Insight:** The dataset consists of **250 job postings** with a mix of numerical and categorical features. While generally clean, key areas like `skills` (20% missing) and `experience_required` (17.2% missing) required specific handling to ensure analytical integrity.

### 2. **Salary Distribution & Correlation**
> **Insight:** There is a **96.3% correlation** between minimum and maximum salaries, indicating very consistent salary banding. Most roles fall within the **$84k - $180k (Min)** and **$169k - $366k (Max)** ranges, showing a high-compensation market.

### 3. **The "Tech Lead" Demand**
> **Insight:** **Engineering Manager, Senior Software Engineer, and Lead Engineer** are the most frequent titles. This highlights a market heavily skewed toward **senior-level technology leadership** rather than entry-level roles.

### 4. **Geographic Hotspots**
> **Insight:** **San Francisco, CA** and **New York, NY** remain the dominant hubs for both job volume and top-tier compensation. A clear positive correlation exists between location-based cost of living and the offered salary floors.

### 5. **Experience vs. Compensation**
> **Insight:** There is a clear **stepped progression** in salary tiers. Entry (0-2y) and Mid (3-5y) levels show steady growth, but a significant "jump" in compensation occurs once a candidate crosses the **6-8 year (Senior)** threshold.

### 6. **In-Demand Skills & Salary Impact**
> **Insight:** **Agile, Git, AWS, and Python** are the most frequently mentioned skills. However, specialized skills like **Machine Learning and Kubernetes** command the highest salary premiums, acting as "high-value" differentiators.

### 7. **Hiring Volume by Company Size**
> **Insight:** The market is a healthy mix of small-to-medium enterprises and "Large" hiring leaders (7+ postings). Interestingly, **Medium-sized companies** often offer the most competitive salary ranges to attract talent away from larger corporations.

## 🛠️ **Technical Stack**
---
* **Data Wrangling:** `Pandas`, `NumPy`
* **Visualization:** `Plotly Express` & `Graph Objects` (Interactive), `Seaborn`, `Matplotlib`
* **Statistics:** `Scipy` (ANOVA Testing for Salary Significance)
* **Machine Learning (Ready):** `Scikit-Learn` (Random Forest & Linear Regression)

## 🚀 **How to Run**
---
1.  **Environment:** Ensure you have a virtual environment active.
2.  **Dependencies:** Install via `pip install pandas numpy matplotlib seaborn plotly sklearn scipy`
3.  **Data:** Place `job_market.csv` in the root directory.
4.  **Execute:** Run `python job_analysis.py`

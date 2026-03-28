import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

# For ML modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import re
from collections import Counter

print("✅ All libraries imported successfully!")

# Load the dataset
df = pd.read_csv('job_market.csv')

# Basic dataset information
print("📊 DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# Display column information
print("\n📋 COLUMN INFORMATION:")
print("=" * 30)
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    non_null = df[col].notna().sum()
    null_count = df[col].isna().sum()
    print(f"{i+1:2d}. {col:<20} | {str(dtype):<10} | {non_null:3d} non-null | {null_count:2d} nulls")

print("\n🔍 FIRST 5 ROWS:")
print("=" * 20)
df.head()

fig = go.Figure()

missing_data = df.isnull().sum()
missing_pct = (missing_data / len(df)) * 100

fig.add_trace(go.Bar(
    x=missing_pct.index.astype(str),
    y=missing_pct.values,
    text=[f'{pct:.1f}%' for pct in missing_pct.values],
    textposition='auto',
    marker_color=['#27ae60' if pct == 0 else '#e74c3c' for pct in missing_pct.values],
    hovertemplate='<b>%{x}</b><br>Missing: %{y:.1f}%<extra></extra>'
))

fig.update_layout(
    title={
        'text': 'Data Completeness Analysis',
        'x': 0.5,
        'font': {'size': 20}
    },
    xaxis_title='Columns',
    yaxis_title='Missing Values (%)',
    template='plotly_white',
    height=500,
    font=dict(size=12),
    xaxis=dict(tickangle=45)
)

fig.show()

"""
Interpretation
Dataset Structure: We have 250 job postings with 10 features
Data Completeness: Most columns have complete data, but job_type (12% missing), category (8% missing), experience_required (17.2% missing), and skills (20% missing) have some missing values
Data Types: Mix of numerical (3) and categorical (7) features
Key Insights: The dataset is relatively clean with most missing values in non-critical columns
"""

# Salary statistics
print("💰 SALARY STATISTICS")
print("=" * 40)
print("Salary Min:")
print(df['salary_min'].describe())
print("\nSalary Max:")
print(df['salary_max'].describe())

# Calculate salary range
df['salary_range'] = df['salary_max'] - df['salary_min']
print(f"\n📊 Salary Range Statistics:")
print(f"Mean Range: ${df['salary_range'].mean():,.0f}")
print(f"Median Range: ${df['salary_range'].median():,.0f}")
print(f"Min Range: ${df['salary_range'].min():,.0f}")
print(f"Max Range: ${df['salary_range'].max():,.0f}")

# Create salary distribution visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Salary Min Distribution', 'Salary Max Distribution', 
                   'Salary Range Distribution', 'Min vs Max Salary Correlation'),
    specs=[[{'secondary_y': False}, {'secondary_y': False}],
           [{'secondary_y': False}, {'secondary_y': False}]]
)

# Salary Min histogram
fig.add_trace(
    go.Histogram(x=df['salary_min'], nbinsx=30, name='Salary Min', 
                marker_color='#3498db', opacity=0.7),
    row=1, col=1
)

# Salary Max histogram
fig.add_trace(
    go.Histogram(x=df['salary_max'], nbinsx=30, name='Salary Max', 
                marker_color='#e74c3c', opacity=0.7),
    row=1, col=2
)

# Salary Range histogram
fig.add_trace(
    go.Histogram(x=df['salary_range'], nbinsx=30, name='Salary Range', 
                marker_color='#27ae60', opacity=0.7),
    row=2, col=1
)

# Scatter plot Min vs Max
fig.add_trace(
    go.Scatter(x=df['salary_min'], y=df['salary_max'], mode='markers',
              marker=dict(color='#9b59b6', size=8, opacity=0.6),
              name='Min vs Max', showlegend=False),
    row=2, col=2
)

fig.update_layout(
    title_text="💰 Comprehensive Salary Analysis",
    title_x=0.5,
    height=800,
    showlegend=True,
    template='plotly_white'
)

# Update axes labels
fig.update_xaxes(title_text="Salary (USD)", row=1, col=1)
fig.update_xaxes(title_text="Salary (USD)", row=1, col=2)
fig.update_xaxes(title_text="Range (USD)", row=2, col=1)
fig.update_xaxes(title_text="Min Salary (USD)", row=2, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=2)
fig.update_yaxes(title_text="Frequency", row=2, col=1)
fig.update_yaxes(title_text="Max Salary (USD)", row=2, col=2)

fig.show()

# Calculate correlation
correlation = df['salary_min'].corr(df['salary_max'])
print(f"📈 Correlation between Min and Max Salary: {correlation:.3f}")


"""
Correlation between Min and Max Salary: 0.963
📌 Interpretation
Salary Ranges: Minimum salaries range from ~84k to 180K, while maximum salaries range from ~ 169k to 366K
Distribution: Both salary_min and salary_max show roughly normal distributions with some right skew
Correlation: Strong positive correlation (0.85+) between min and max salaries, indicating consistent salary banding
Salary Bands: Most jobs have ranges between 50k to 150K, suggesting competitive market positioning

"""

# Job title analysis
job_title_counts = df['job_title'].value_counts()
print("🏆 TOP 10 MOST COMMON JOB TITLES")
print("=" * 40)
for i, (title, count) in enumerate(job_title_counts.head(10).items(), 1):
    percentage = (count / len(df)) * 100
    avg_salary_min = df[df['job_title'] == title]['salary_min'].mean()
    avg_salary_max = df[df['job_title'] == title]['salary_max'].mean()
    print(f"{i:2d}. {title:<25} | {count:2d} jobs ({percentage:5.1f}%) | Avg: ${avg_salary_min:6.0f} - ${avg_salary_max:6.0f}")

# Calculate average salary by job title
salary_by_title = df.groupby('job_title').agg({
    'salary_min': ['mean', 'median', 'count'],
    'salary_max': ['mean', 'median'],
    'experience_required': 'mean'
}).round(0)

salary_by_title.columns = ['min_mean', 'min_median', 'count', 'max_mean', 'max_median', 'avg_exp']
salary_by_title = salary_by_title[salary_by_title['count'] >= 2].sort_values('min_mean', ascending=False)

print(f"\n💰 TOP PAYING JOB TITLES (min 2 positions):")
print("=" * 50)
for i, (title, row) in enumerate(salary_by_title.head(10).iterrows(), 1):
    print(f"{i:2d}. {title:<25} | ${row['min_mean']:6.0f} - ${row['max_mean']:6.0f} | {row['count']:2.0f} positions")


# Create job title visualization
# Get top 15 job titles for better visualization
top_titles = job_title_counts.head(15)

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Top 15 Job Titles by Frequency', 'Average Salary by Job Title'),
    vertical_spacing=0.12
)

# Job frequency bar chart
fig.add_trace(
    go.Bar(x=top_titles.index, y=top_titles.values,
           marker_color='#3498db', name='Job Count',
           text=top_titles.values, textposition='auto'),
    row=1, col=1
)

# Average salary by job title (for titles with 2+ positions)
salary_plot_data = salary_by_title.head(10)
fig.add_trace(
    go.Scatter(x=salary_plot_data.index, y=salary_plot_data['min_mean'],
               mode='markers+lines', name='Avg Min Salary',
               marker=dict(size=10, color='#e74c3c'), line=dict(width=3)),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=salary_plot_data.index, y=salary_plot_data['max_mean'],
               mode='markers+lines', name='Avg Max Salary',
               marker=dict(size=10, color='#27ae60'), line=dict(width=3)),
    row=2, col=1
)

fig.update_layout(
    title_text="🏢 Job Title Analysis Dashboard",
    title_x=0.5,
    height=900,
    template='plotly_white',
    showlegend=True
)

fig.update_xaxes(title_text="Job Title", row=1, col=1)
fig.update_yaxes(title_text="Number of Positions", row=1, col=1)
fig.update_xaxes(title_text="Job Title", row=2, col=1)
fig.update_yaxes(title_text="Average Salary (USD)", row=2, col=1)

fig.update_xaxes(tickangle=45, row=1, col=1)
fig.update_xaxes(tickangle=45, row=2, col=1)

fig.show()


"""
Interpretation¶
Most Common Roles: Engineering Manager, Senior Software Engineer, and Lead Engineer dominate the job market
Salary Variations: Different job titles show significant salary differences, with management positions commanding higher pay
Market Demand: Technology leadership roles are in high demand based on frequency
Compensation Tiers: Clear salary tiers emerge based on seniority and role complexity
"""

# Location analysis
location_counts = df['location'].value_counts()
print("🌍 TOP 10 LOCATIONS BY JOB COUNT")
print("=" * 40)

# Calculate salary statistics by location
location_salary = df.groupby('location').agg({
    'salary_min': ['mean', 'median', 'count'],
    'salary_max': ['mean', 'median'],
    'experience_required': 'mean'
}).round(0)

location_salary.columns = ['min_mean', 'min_median', 'count', 'max_mean', 'max_median', 'avg_exp']
location_salary = location_salary[location_salary['count'] >= 3].sort_values('min_mean', ascending=False)

for i, (location, row) in enumerate(location_salary.head(10).iterrows(), 1):
    print(f"{i:2d}. {location:<20} | {row['count']:2.0f} jobs | ${row['min_mean']:6.0f} - ${row['max_mean']:6.0f}")

# Extract state information
df['state'] = df['location'].str.extract(r', ([A-Z]{2})$')
state_analysis = df.groupby('state').agg({
    'salary_min': 'mean',
    'salary_max': 'mean',
    'job_title': 'count'
}).round(0)

state_analysis = state_analysis[state_analysis['job_title'] >= 5].sort_values('salary_min', ascending=False)
print(f"\n🗺️ SALARY BY STATE (min 5 jobs):")
print("=" * 35)
for state, row in state_analysis.iterrows():
    print(f"{state}: ${row['salary_min']:6.0f} - ${row['salary_max']:6.0f} ({row['job_title']:2.0f} jobs)")

# Create location visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Top Cities by Job Count', 'Average Salary by City', 
                   'Salary Distribution by Top States', 'Experience vs Salary by Location'),
    specs=[[{'secondary_y': False}, {'secondary_y': False}],
           [{'secondary_y': False}, {'secondary_y': False}]]
)

# Top cities by job count
top_cities = location_counts.head(10)
fig.add_trace(
    go.Bar(x=top_cities.values, y=top_cities.index, orientation='h',
           marker_color='#3498db', name='Job Count',
           text=top_cities.values, textposition='auto'),
    row=1, col=1
)

# Average salary by city (top 8 cities with 3+ jobs)
top_salary_cities = location_salary.head(8)
fig.add_trace(
    go.Scatter(x=top_salary_cities['min_mean'], y=top_salary_cities.index,
               mode='markers', name='Avg Min Salary',
               marker=dict(size=12, color='#e74c3c')),
    row=1, col=2
)

# State salary comparison
if len(state_analysis) > 0:
    fig.add_trace(
        go.Bar(x=state_analysis.index, y=state_analysis['salary_min'],
               name='Min Salary', marker_color='#27ae60', opacity=0.8),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=state_analysis.index, y=state_analysis['salary_max'],
               name='Max Salary', marker_color='#f39c12', opacity=0.8),
        row=2, col=1
    )

# Experience vs Salary for top cities
top_cities_list = location_counts.head(5).index.tolist()
for i, city in enumerate(top_cities_list):
    city_data = df[df['location'] == city]
    colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6']
    fig.add_trace(
        go.Scatter(x=city_data['experience_required'], y=city_data['salary_min'],
                   mode='markers', name=city,
                   marker=dict(size=8, color=colors[i], opacity=0.7)),
        row=2, col=2
    )

fig.update_layout(
    title_text="📍 Geographic Salary Analysis",
    title_x=0.5,
    height=1000,
    template='plotly_white',
    showlegend=True
)

fig.update_xaxes(title_text="Number of Jobs", row=1, col=1)
fig.update_yaxes(title_text="City", row=1, col=1)
fig.update_xaxes(title_text="Average Min Salary (USD)", row=1, col=2)
fig.update_yaxes(title_text="City", row=1, col=2)
fig.update_xaxes(title_text="State", row=2, col=1)
fig.update_yaxes(title_text="Salary (USD)", row=2, col=1)
fig.update_xaxes(title_text="Experience Required (years)", row=2, col=2)
fig.update_yaxes(title_text="Min Salary (USD)", row=2, col=2)

fig.show()

"""
 Interpretation
Geographic Distribution: San Francisco, CA and New York, NY dominate the job market
Salary by Location: Significant variations exist, with SF and NY showing higher average salaries
Market Concentration: Major tech hubs (SF, Seattle, NY) have the highest job concentration
Experience-Salary Relationship: Clear positive correlation between experience required and salary across all locations
"""

# Job type analysis
job_type_counts = df['job_type'].value_counts()
print("💼 JOB TYPE DISTRIBUTION")
print("=" * 30)
for job_type, count in job_type_counts.items():
    percentage = (count / df['job_type'].notna().sum()) * 100
    avg_salary = df[df['job_type'] == job_type]['salary_min'].mean()
    print(f"{job_type:<15}: {count:3d} jobs ({percentage:5.1f}%) | Avg Salary: ${avg_salary:6.0f}")

# Category analysis
category_counts = df['category'].value_counts()
print(f"\n📂 JOB CATEGORY DISTRIBUTION")
print("=" * 30)
for category, count in category_counts.items():
    percentage = (count / df['category'].notna().sum()) * 100
    avg_salary = df[df['category'] == category]['salary_min'].mean()
    print(f"{category:<15}: {count:3d} jobs ({percentage:5.1f}%) | Avg Salary: ${avg_salary:6.0f}")

# Statistical analysis
from scipy import stats

# Test if job types have significantly different salaries
job_type_groups = [df[df['job_type'] == jt]['salary_min'].dropna() for jt in job_type_counts.index]
f_stat, p_value = stats.f_oneway(*job_type_groups)

print(f"\n📊 STATISTICAL ANALYSIS:")
print(f"ANOVA F-statistic: {f_stat:.3f}")
print(f"P-value: {p_value:.6f}")
print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

# Create job type and category visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Job Type Distribution', 'Job Category Distribution', 
                   'Salary by Job Type', 'Salary by Job Category'),
    specs=[[{'type': 'pie'}, {'type': 'pie'}],
           [{'secondary_y': False}, {'secondary_y': False}]]
)

# Job type pie chart
fig.add_trace(
    go.Pie(labels=job_type_counts.index, values=job_type_counts.values,
           name="Job Type", hole=0.4,
           marker_colors=['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6']),
    row=1, col=1
)

# Category pie chart
fig.add_trace(
    go.Pie(labels=category_counts.index, values=category_counts.values,
           name="Category", hole=0.4,
           marker_colors=['#e74c3c', '#3498db', '#27ae60', '#f39c12']),
    row=1, col=2
)

# Box plot for salary by job type
for i, job_type in enumerate(job_type_counts.index):
    job_type_data = df[df['job_type'] == job_type]['salary_min'].dropna()
    colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6']
    fig.add_trace(
        go.Box(y=job_type_data, name=job_type,
               marker_color=colors[i % len(colors)]),
        row=2, col=1
    )

# Box plot for salary by category
for i, category in enumerate(category_counts.index):
    category_data = df[df['category'] == category]['salary_min'].dropna()
    colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12']
    fig.add_trace(
        go.Box(y=category_data, name=category,
               marker_color=colors[i % len(colors)]),
        row=2, col=2
    )

fig.update_layout(
    title_text="💼 Job Type & Category Analysis",
    title_x=0.5,
    height=1000,
    template='plotly_white',
    showlegend=False
)

fig.update_yaxes(title_text="Salary (USD)", row=2, col=1)
fig.update_yaxes(title_text="Salary (USD)", row=2, col=2)

fig.show()

"""
 Interpretation
Job Type Distribution: Full-time positions dominate (65%+), followed by Remote and Part-time
Category Focus: Technology sector represents the vast majority of positions
Salary Variations: Full-time positions tend to have higher average salaries than part-time
Market Structure: The job market is heavily concentrated in tech with full-time employment

"""

# Experience analysis
exp_stats = df['experience_required'].describe()
print("📊 EXPERIENCE REQUIREMENT STATISTICS")
print("=" * 40)
print(f"Mean Experience: {exp_stats['mean']:.1f} years")
print(f"Median Experience: {exp_stats['50%']:.1f} years")
print(f"Min Experience: {exp_stats['min']:.0f} years")
print(f"Max Experience: {exp_stats['max']:.0f} years")

# Create experience bins
df['exp_category'] = pd.cut(df['experience_required'], 
                           bins=[0, 2, 5, 8, 15], 
                           labels=['Entry (0-2y)', 'Mid (3-5y)', 'Senior (6-8y)', 'Lead (9+y)'])

exp_salary_analysis = df.groupby('exp_category').agg({
    'salary_min': ['mean', 'median', 'count'],
    'salary_max': ['mean', 'median']
}).round(0)

exp_salary_analysis.columns = ['min_mean', 'min_median', 'count', 'max_mean', 'max_median']

print(f"\n💰 SALARY BY EXPERIENCE LEVEL:")
print("=" * 35)
for level, row in exp_salary_analysis.iterrows():
    if pd.notna(level):
        print(f"{level:<15}: ${row['min_mean']:6.0f} - ${row['max_mean']:6.0f} ({row['count']:2.0f} jobs)")

# Correlation analysis
exp_salary_corr = df['experience_required'].corr(df['salary_min'])
exp_max_corr = df['experience_required'].corr(df['salary_max'])
print(f"\n📈 CORRELATION ANALYSIS:")
print(f"Experience vs Min Salary: {exp_salary_corr:.3f}")
print(f"Experience vs Max Salary: {exp_max_corr:.3f}")

#Create experience analysis visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Experience Distribution', 'Salary by Experience Level', 
                   'Experience vs Salary Correlation', 'Experience Requirements by Job Title'),
    specs=[[{'secondary_y': False}, {'secondary_y': False}],
           [{'secondary_y': False}, {'secondary_y': False}]]
)

# Experience histogram
fig.add_trace(
    go.Histogram(x=df['experience_required'], nbinsx=15,
                marker_color='#3498db', name='Experience Distribution',
                opacity=0.7),
    row=1, col=1
)

# Salary by experience category
exp_categories = exp_salary_analysis.index.dropna()
fig.add_trace(
    go.Bar(x=exp_categories, y=exp_salary_analysis.loc[exp_categories, 'min_mean'],
           name='Min Salary', marker_color='#e74c3c', opacity=0.8),
    row=1, col=2
)

fig.add_trace(
    go.Bar(x=exp_categories, y=exp_salary_analysis.loc[exp_categories, 'max_mean'],
           name='Max Salary', marker_color='#27ae60', opacity=0.8),
    row=1, col=2
)

# Experience vs Salary scatter
fig.add_trace(
    go.Scatter(x=df['experience_required'], y=df['salary_min'],
               mode='markers', name='Min Salary',
               marker=dict(size=8, color='#9b59b6', opacity=0.6)),
    row=2, col=1
)

# Add trend line
z = np.polyfit(df['experience_required'].dropna(), 
               df[df['experience_required'].notna()]['salary_min'], 1)
p = np.poly1d(z)
x_trend = np.linspace(df['experience_required'].min(), df['experience_required'].max(), 100)

fig.add_trace(
    go.Scatter(x=x_trend, y=p(x_trend),
               mode='lines', name='Trend Line',
               line=dict(color='#f39c12', width=3)),
    row=2, col=1
)

# Experience by job title (top 5 titles)
top_job_titles = df['job_title'].value_counts().head(5).index
for i, title in enumerate(top_job_titles):
    title_data = df[df['job_title'] == title]['experience_required'].dropna()
    colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6']
    fig.add_trace(
        go.Box(y=title_data, name=title,
               marker_color=colors[i], showlegend=False),
        row=2, col=2
    )

fig.update_layout(
    title_text="📈 Experience Impact Analysis",
    title_x=0.5,
    height=1000,
    template='plotly_white',
    showlegend=True
)

fig.update_xaxes(title_text="Experience (years)", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_xaxes(title_text="Experience Level", row=1, col=2)
fig.update_yaxes(title_text="Salary (USD)", row=1, col=2)
fig.update_xaxes(title_text="Experience (years)", row=2, col=1)
fig.update_yaxes(title_text="Min Salary (USD)", row=2, col=1)
fig.update_yaxes(title_text="Experience (years)", row=2, col=2)

fig.show()

"""
 Interpretation¶
Experience Distribution: Most positions require 3-8 years of experience, with a peak around 4-6 years
Strong Correlation: Experience shows moderate positive correlation (~0.4-0.5) with salary levels
Salary Progression: Clear salary tiers based on experience levels (Entry < Mid < Senior < Lead)
Experience by Role: Different job titles show distinct experience requirement patterns
"""

# Skills analysis
import re
from collections import Counter

# Extract all skills
all_skills = []
for skills_str in df['skills'].dropna():
    if isinstance(skills_str, str):
        skills_list = [skill.strip() for skill in skills_str.split(',')]
        all_skills.extend(skills_list)

# Count skill frequency
skill_counts = Counter(all_skills)
top_skills = dict(skill_counts.most_common(20))

print("🔥 TOP 20 MOST IN-DEMAND SKILLS")
print("=" * 40)
for i, (skill, count) in enumerate(top_skills.items(), 1):
    percentage = (count / len(df['skills'].dropna())) * 100
    print(f"{i:2d}. {skill:<20}: {count:3d} mentions ({percentage:5.1f}%)")

# Calculate average salary for top skills
skill_salary_impact = {}
for skill in list(top_skills.keys())[:10]:
    skill_jobs = df[df['skills'].str.contains(skill, na=False)]
    if len(skill_jobs) >= 5:  # At least 5 jobs
        avg_salary = skill_jobs['salary_min'].mean()
        skill_salary_impact[skill] = {
            'count': len(skill_jobs),
            'avg_salary': avg_salary,
            'salary_range': skill_jobs['salary_max'].mean() - avg_salary
        }

print(f"\n💰 SALARY IMPACT OF TOP SKILLS:")
print("=" * 40)
for skill, data in sorted(skill_salary_impact.items(), key=lambda x: x[1]['avg_salary'], reverse=True):
    print(f"{skill:<20}: ${data['avg_salary']:6.0f} avg ({data['count']:2d} jobs)")

# Create skills visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Top 15 Skills by Demand', 'Salary Impact of Top Skills', 
                   'Skills vs Experience Heatmap', 'Skill Category Distribution'),
    specs=[[{'secondary_y': False}, {'secondary_y': False}],
           [{'secondary_y': False}, {'type': 'pie'}]]
)

# Top skills bar chart
top_15_skills = dict(list(top_skills.items())[:15])
fig.add_trace(
    go.Bar(x=list(top_15_skills.values()), y=list(top_15_skills.keys()),
           orientation='h', marker_color='#3498db',
           text=list(top_15_skills.values()), textposition='auto'),
    row=1, col=1
)

# Salary impact of skills
if skill_salary_impact:
    skills_list = list(skill_salary_impact.keys())
    salaries = [skill_salary_impact[skill]['avg_salary'] for skill in skills_list]
    counts = [skill_salary_impact[skill]['count'] for skill in skills_list]
    
    fig.add_trace(
        go.Scatter(x=counts, y=salaries, mode='markers+text',
                   text=skills_list, textposition='top center',
                   marker=dict(size=12, color='#e74c3c', opacity=0.7),
                   name='Skill Salary Impact'),
        row=1, col=2
    )

# Skills heatmap (top 10 skills vs experience)
top_10_skills = list(top_skills.keys())[:10]
heatmap_data = []
skill_labels = []

for skill in top_10_skills:
    skill_jobs = df[df['skills'].str.contains(skill, na=False)]
    if len(skill_jobs) >= 3:
        avg_exp = skill_jobs['experience_required'].mean()
        avg_salary = skill_jobs['salary_min'].mean()
        heatmap_data.append([avg_exp, avg_salary])
        skill_labels.append(skill)

if heatmap_data:
    heatmap_array = np.array(heatmap_data)
    fig.add_trace(
        go.Scatter(x=heatmap_array[:, 0], y=heatmap_array[:, 1],
                   mode='markers+text', text=skill_labels,
                   textposition='top center',
                   marker=dict(size=15, color=heatmap_array[:, 1], 
                              colorscale='Viridis', showscale=True),
                   name='Skills Heatmap'),
        row=2, col=1
    )

# Skill categories pie chart
skill_categories = {
    'Programming': ['Python', 'Java', 'JavaScript', 'Ruby', 'Go', 'TypeScript', 'SQL', 'C++'],
    'Cloud/DevOps': ['AWS', 'Azure', 'Kubernetes', 'Docker'],
    'Frameworks': ['React', 'Angular', 'Node.js', 'Django'],
    'Databases': ['MongoDB', 'PostgreSQL', 'MySQL', 'Redis'],
    'Tools': ['Git', 'Agile', 'REST APIs', 'GraphQL'],
    'Other': ['Machine Learning', 'TensorFlow', 'Linux', 'Microservices']
}

category_counts = {}
for category, skills in skill_categories.items():
    count = sum(skill_counts.get(skill, 0) for skill in skills)
    if count > 0:
        category_counts[category] = count

fig.add_trace(
    go.Pie(labels=list(category_counts.keys()), values=list(category_counts.values()),
           name="Skill Categories", hole=0.4,
           marker_colors=['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#e67e22']),
    row=2, col=2
)

fig.update_layout(
    title_text="🛠️ Skills Analysis Dashboard",
    title_x=0.5,
    height=1200,
    template='plotly_white',
    showlegend=True
)

fig.update_xaxes(title_text="Demand Count", row=1, col=1)
fig.update_yaxes(title_text="Skills", row=1, col=1)
fig.update_xaxes(title_text="Job Count", row=1, col=2)
fig.update_yaxes(title_text="Average Salary (USD)", row=1, col=2)
fig.update_xaxes(title_text="Avg Experience (years)", row=2, col=1)
fig.update_yaxes(title_text="Avg Min Salary (USD)", row=2, col=1)

fig.show()

"""
 Interpretation
Top Skills: Agile, Git, AWS, Python, and JavaScript are the most demanded skills
Skill Categories: Programming languages dominate, followed by Cloud/DevOps tools
Salary Impact: Certain skills (Machine Learning, AWS, Kubernetes) command premium salaries
Market Trends: Modern tech stack emphasizes cloud, containers, and modern programming languages
"""

# Company analysis
company_counts = df['company'].value_counts()
print("🏢 TOP 15 COMPANIES BY HIRING VOLUME")
print("=" * 45)

company_salary_analysis = df.groupby('company').agg({
    'salary_min': ['mean', 'median', 'count'],
    'salary_max': ['mean', 'median'],
    'experience_required': 'mean'
}).round(0)

company_salary_analysis.columns = ['min_mean', 'min_median', 'count', 'max_mean', 'max_median', 'avg_exp']
company_salary_analysis = company_salary_analysis[company_salary_analysis['count'] >= 2].sort_values('min_mean', ascending=False)

for i, (company, row) in enumerate(company_salary_analysis.head(15).iterrows(), 1):
    print(f"{i:2d}. {company:<20} | {row['count']:2.0f} jobs | ${row['min_mean']:6.0f} - ${row['max_mean']:6.0f} | Exp: {row['avg_exp']:4.1f}y")

# Company size analysis (based on job postings)
company_size_categories = pd.cut(company_counts.values, 
                                bins=[0, 1, 3, 6, float('inf')], 
                                labels=['Single Post', 'Small (2-3)', 'Medium (4-6)', 'Large (7+)'])

size_distribution = company_size_categories.value_counts()
print(f"\n📊 COMPANY SIZE DISTRIBUTION:")
print("=" * 35)
for size, count in size_distribution.items():
    percentage = (count / len(company_counts)) * 100
    print(f"{size:<15}: {count:3d} companies ({percentage:5.1f}%)")

# Calculate average salary by company size
size_salary_map = {}
for company, count in company_counts.items():
    if count == 1:
        size = 'Single Post'
    elif count <= 3:
        size = 'Small (2-3)'
    elif count <= 6:
        size = 'Medium (4-6)'
    else:
        size = 'Large (7+)'
    
    company_salary = df[df['company'] == company]['salary_min'].mean()
    if size not in size_salary_map:
        size_salary_map[size] = []
    size_salary_map[size].append(company_salary)

print(f"\n💰 SALARY BY COMPANY SIZE:")
print("=" * 30)
for size, salaries in size_salary_map.items():
    avg_salary = np.mean(salaries)
    print(f"{size:<15}: ${avg_salary:6.0f} average")

# Create company analysis visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Top 12 Companies by Hiring Volume', 'Average Salary by Company', 
                   'Company Size Distribution', 'Salary Distribution by Company Size'),
    specs=[[{'secondary_y': False}, {'secondary_y': False}],
           [{'type': 'pie'}, {'secondary_y': False}]]
)

# Top companies by hiring volume
top_companies = company_counts.head(12)
fig.add_trace(
    go.Bar(x=top_companies.values, y=top_companies.index, orientation='h',
           marker_color='#3498db', name='Job Count',
           text=top_companies.values, textposition='auto'),
    row=1, col=1
)

# Average salary by company (companies with 2+ jobs)
top_salary_companies = company_salary_analysis.head(10)
fig.add_trace(
    go.Scatter(x=top_salary_companies['min_mean'], y=top_salary_companies.index,
               mode='markers', name='Avg Min Salary',
               marker=dict(size=top_salary_companies['count']*3, 
                          color='#e74c3c', opacity=0.7,
                          line=dict(width=2, color='darkred'))),
    row=1, col=2
)

# Company size distribution pie chart
fig.add_trace(
    go.Pie(labels=size_distribution.index, values=size_distribution.values,
           name="Company Size", hole=0.4,
           marker_colors=['#3498db', '#e74c3c', '#27ae60', '#f39c12']),
    row=2, col=1
)

# Salary by company size
size_salary_data = [(size, np.mean(salaries)) for size, salaries in size_salary_map.items()]
size_salary_data.sort(key=lambda x: x[1], reverse=True)

sizes = [item[0] for item in size_salary_data]
salaries = [item[1] for item in size_salary_data]

fig.add_trace(
    go.Bar(x=sizes, y=salaries,
           marker_color=['#e74c3c', '#3498db', '#27ae60', '#f39c12'],
           name='Avg Salary', text=[f'${s:,.0f}' for s in salaries],
           textposition='auto'),
    row=2, col=2
)

fig.update_layout(
    title_text="🏢 Company Analysis Dashboard",
    title_x=0.5,
    height=1000,
    template='plotly_white',
    showlegend=True
)

fig.update_xaxes(title_text="Number of Jobs", row=1, col=1)
fig.update_yaxes(title_text="Company", row=1, col=1)
fig.update_xaxes(title_text="Average Min Salary (USD)", row=1, col=2)
fig.update_yaxes(title_text="Company", row=1, col=2)
fig.update_yaxes(title_text="Average Salary (USD)", row=2, col=2)

fig.show()

"""
 Interpretation¶
Hiring Leaders: DataInc, EnterpriseHub, and StartupXYZ are the top hiring companies
Company Size: Most companies are small to medium-sized (single to 6 postings)
Salary Patterns: Larger companies (7+ postings) tend to offer competitive salaries
Market Distribution: Healthy mix of company sizes, indicating diverse employment opportunities
"""

# Missing values analysis
missing_summary = df.isnull().sum()
missing_percentage = (missing_summary / len(df)) * 100

print("❌ MISSING VALUES ANALYSIS")
print("=" * 40)
print(f"{'Column':<20} {'Missing':<10} {'Percentage':<10}")
print("-" * 40)
for col in df.columns:
    missing_count = missing_summary[col]
    missing_pct = missing_percentage[col]
    if missing_count > 0:
        print(f"{col:<20} {missing_count:<10} {missing_pct:>8.1f}%")
    else:
        print(f"{col:<20} {missing_count:<10} {missing_pct:>8.1f}% ✓")

# Analyze patterns in missing data
print(f"\n📈 MISSING DATA PATTERNS:")
print("=" * 25)

# Check if missing values are correlated
missing_corr = df.isnull().corr()
high_missing_corr = missing_corr[missing_corr > 0.3].stack().drop_duplicates()

print("High correlation in missing values (>0.3):")
for (col1, col2), corr in high_missing_corr.items():
    if col1 != col2 and corr > 0.3:
        print(f"{col1} - {col2}: {corr:.3f}")

# Impact analysis
print(f"\n💥 IMPACT ON ANALYSIS:")
print("=" * 22)
for col in ['job_type', 'category', 'experience_required', 'skills']:
    missing_pct = missing_percentage[col]
    if missing_pct > 0:
        impact = "High" if missing_pct > 15 else "Medium" if missing_pct > 5 else "Low"
        print(f"{col}: {missing_pct:.1f}% missing - {impact} impact")

# Create missing values visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Missing Values by Column', 'Data Completeness Heatmap', 
                   'Missing Values Impact on Salary', 'Missing vs Non-Missing Comparison'),
    specs=[[{'secondary_y': False}, {'secondary_y': False}],
           [{'secondary_y': False}, {'secondary_y': False}]]
)

# Missing values bar chart
missing_data = missing_percentage[missing_percentage > 0]
fig.add_trace(
    go.Bar(x=missing_data.index, y=missing_data.values,
           marker_color=['#e74c3c' if pct > 15 else '#f39c12' if pct > 5 else '#27ae60' for pct in missing_data.values],
           text=[f'{pct:.1f}%' for pct in missing_data.values],
           textposition='auto', name='Missing %'),
    row=1, col=1
)

# Data completeness heatmap
completeness_data = (1 - missing_percentage / 100).values.reshape(1, -1)
fig.add_trace(
    go.Heatmap(z=completeness_data,
               x=df.columns,
               colorscale='RdYlGn',
               showscale=True,
               text=[[f'{val:.1%}' for val in (1 - missing_percentage / 100)]],
               texttemplate="%{text}",
               textfont={"size": 10}),
    row=1, col=2
)

# Impact on salary analysis
salary_with_skills = df[df['skills'].notna()]['salary_min'].mean()
salary_without_skills = df[df['skills'].isna()]['salary_min'].mean()

salary_with_exp = df[df['experience_required'].notna()]['salary_min'].mean()
salary_without_exp = df[df['experience_required'].isna()]['salary_min'].mean()

impact_data = {
    'Skills Info': [salary_with_skills, salary_without_skills],
    'Experience Info': [salary_with_exp, salary_without_exp]
}

categories = ['With Info', 'Without Info']
colors = ['#27ae60', '#e74c3c']

for i, (category, values) in enumerate(impact_data.items()):
    fig.add_trace(
        go.Bar(x=categories, y=values, name=category,
               marker_color=colors, opacity=0.8,
               text=[f'${v:,.0f}' for v in values],
               textposition='auto'),
        row=2, col=1
    )
    break  # Only show one comparison to avoid clutter

# Comparison of complete vs incomplete records
complete_records = df.dropna()
incomplete_records = df[df.isnull().any(axis=1)]

comparison_data = {
    'Complete Records': [len(complete_records), complete_records['salary_min'].mean()],
    'Incomplete Records': [len(incomplete_records), incomplete_records['salary_min'].mean()]
}

fig.add_trace(
    go.Scatter(x=[comparison_data['Complete Records'][0]], 
               y=[comparison_data['Complete Records'][1]],
               mode='markers', name='Complete Records',
               marker=dict(size=20, color='#27ae60')),
    row=2, col=2
)

fig.add_trace(
    go.Scatter(x=[comparison_data['Incomplete Records'][0]], 
               y=[comparison_data['Incomplete Records'][1]],
               mode='markers', name='Incomplete Records',
               marker=dict(size=20, color='#e74c3c')),
    row=2, col=2
)

fig.update_layout(
    title_text="📊 Data Quality & Missing Values Analysis",
    title_x=0.5,
    height=1000,
    template='plotly_white',
    showlegend=True
)

fig.update_yaxes(title_text="Missing %", row=1, col=1)
fig.update_yaxes(title_text="Completeness", row=1, col=2)
fig.update_yaxes(title_text="Average Salary (USD)", row=2, col=1)
fig.update_xaxes(title_text="Record Count", row=2, col=2)
fig.update_yaxes(title_text="Average Salary (USD)", row=2, col=2)

fig.update_xaxes(tickangle=45, row=1, col=1)
fig.show()

"""
 Interpretation¶
Missing Data Pattern: Skills (20%) and experience_required (17.2%) have the most missing values
Impact Assessment: Missing skills data doesn't significantly affect salary averages
Data Quality: Overall 85%+ completeness across all columns
Completeness vs Salary: Records with complete information show slightly higher average salaries
"""

# Outlier detection using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect outliers in salary_min and salary_max
outliers_min, lower_min, upper_min = detect_outliers_iqr(df, 'salary_min')
outliers_max, lower_max, upper_max = detect_outliers_iqr(df, 'salary_max')

print("🔍 OUTLIER DETECTION RESULTS")
print("=" * 35)
print(f"Salary Min Outliers: {len(outliers_min)} ({len(outliers_min)/len(df)*100:.1f}%)")
print(f"  - Range: ${lower_min:.0f} - ${upper_min:.0f}")
print(f"  - Outlier values: {sorted(outliers_min['salary_min'].tolist())}")

print(f"\nSalary Max Outliers: {len(outliers_max)} ({len(outliers_max)/len(df)*100:.1f}%)")
print(f"  - Range: ${lower_max:.0f} - ${upper_max:.0f}")
print(f"  - Outlier values: {sorted(outliers_max['salary_max'].tolist())}")

# Z-score method for outlier detection
from scipy import stats

z_scores_min = np.abs(stats.zscore(df['salary_min']))
z_scores_max = np.abs(stats.zscore(df['salary_max']))

outliers_z_min = df[z_scores_min > 2.5]  # Using 2.5 as threshold
outliers_z_max = df[z_scores_max > 2.5]

print(f"\n📊 Z-SCORE OUTLIER DETECTION (|z| > 2.5):")
print(f"Salary Min Z-outliers: {len(outliers_z_min)} ({len(outliers_z_min)/len(df)*100:.1f}%)")
print(f"Salary Max Z-outliers: {len(outliers_z_max)} ({len(outliers_z_max)/len(df)*100:.1f}%)")

# Analyze outlier characteristics
if len(outliers_min) > 0:
    print(f"\n🔍 OUTLIER ANALYSIS:")
    print(f"Low salary outliers characteristics:")
    print(f"  - Avg experience: {outliers_min['experience_required'].mean():.1f} years")
    print(f"  - Most common job title: {outliers_min['job_title'].mode().iloc[0]}")
    print(f"  - Most common location: {outliers_min['location'].mode().iloc[0]}")

if len(outliers_max) > 0:
    print(f"\nHigh salary outliers characteristics:")
    print(f"  - Avg experience: {outliers_max['experience_required'].mean():.1f} years")
    print(f"  - Most common job title: {outliers_max['job_title'].mode().iloc[0]}")
    print(f"  - Most common location: {outliers_max['location'].mode().iloc[0]}")

# Create outlier detection visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Salary Distribution with Outliers', 'Box Plot with Outlier Detection', 
                   'Outlier Characteristics', 'Z-Score Distribution'),
    specs=[[{'secondary_y': False}, {'secondary_y': False}],
           [{'secondary_y': False}, {'secondary_y': False}]]
)

# Salary distribution with outlier boundaries
fig.add_trace(
    go.Histogram(x=df['salary_min'], nbinsx=30, name='Salary Min',
                marker_color='#3498db', opacity=0.7),
    row=1, col=1
)

# Add outlier boundaries
fig.add_vline(x=lower_min, line_dash="dash", line_color="red", 
              annotation_text="Lower Bound", row=1, col=1)
fig.add_vline(x=upper_min, line_dash="dash", line_color="red", 
              annotation_text="Upper Bound", row=1, col=1)

# Box plot for outlier detection
fig.add_trace(
    go.Box(y=df['salary_min'], name='Salary Min',
           marker_color='#e74c3c', boxmean=True),
    row=1, col=2
    )

fig.add_trace(
    go.Box(y=df['salary_max'], name='Salary Max',
           marker_color='#27ae60', boxmean=True),
    row=1, col=2
    )

# Outlier characteristics
if len(outliers_min) > 0 or len(outliers_max) > 0:
    # Combine outliers for analysis
    all_outliers = pd.concat([outliers_min, outliers_max]).drop_duplicates()
    
    fig.add_trace(
        go.Scatter(x=all_outliers['experience_required'], 
                   y=all_outliers['salary_min'],
                   mode='markers', name='Outliers',
                   marker=dict(size=10, color='#f39c12', 
                              symbol='diamond', line=dict(width=2, color='darkorange'))),
        row=2, col=1
    )
    
    # Add normal points for comparison
    normal_data = df[~df.index.isin(all_outliers.index)]
    fig.add_trace(
        go.Scatter(x=normal_data['experience_required'], 
                   y=normal_data['salary_min'],
                   mode='markers', name='Normal Data',
                   marker=dict(size=6, color='#9b59b6', opacity=0.5)),
        row=2, col=1
    )

# Z-score distribution
fig.add_trace(
    go.Histogram(x=z_scores_min, nbinsx=20, name='Z-Score Min',
                marker_color='#9b59b6', opacity=0.7),
    row=2, col=2
)

# Add z-score threshold
fig.add_vline(x=2.5, line_dash="dash", line_color="red", 
              annotation_text="Z=2.5", row=2, col=2)

fig.update_layout(
    title_text="🔍 Outlier Detection & Anomaly Analysis",
    title_x=0.5,
    height=1000,
    template='plotly_white',
    showlegend=True
)

fig.update_xaxes(title_text="Salary (USD)", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Salary (USD)", row=1, col=2)
fig.update_xaxes(title_text="Experience (years)", row=2, col=1)
fig.update_yaxes(title_text="Salary (USD)", row=2, col=1)
fig.update_xaxes(title_text="Z-Score", row=2, col=2)
fig.update_yaxes(title_text="Frequency", row=2, col=2)

fig.show()

"""
 Interpretation
Outlier Detection: IQR method identifies ~5-10% outliers in salary data
Outlier Characteristics: Low outliers often entry-level positions; high outliers are senior roles in expensive locations
Z-Score Analysis: Most data points fall within 2.5 standard deviations
Data Quality: Outliers represent legitimate salary variations, not data errors
"""


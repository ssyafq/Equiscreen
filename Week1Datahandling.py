import pandas as pd

demo = pd.read_sas('data/nhanes/DEMO_J.XPT')
bmx = pd.read_sas('data/nhanes/BMX_J.XPT')
bpx = pd.read_sas('data/nhanes/BPX_J.XPT')

# Check what you've got
#print(demo.shape)

# === STEP 3: Merge DataFrames ===
#SEQN — unique participant ID (this is your merge key)
#RIDAGEYR — age in years
#RIAGENDR — gender (1=Male, 2=Female)
#RIDRETH3 — race/ethnicity (detailed)
#BMXBMI — BMI
#BPXSY1 — systolic blood pressure (1st reading)
#BPXDI1 — diastolic blood pressure (1st reading)


# Merge all three on participant ID
#
df = demo.merge(bmx[['SEQN', 'BMXBMI', 'BMXHT', 'BMXWT']], on='SEQN', how='left')
df = df.merge(bpx[['SEQN', 'BPXSY1', 'BPXDI1']], on='SEQN', how='left')

#print(df.shape)  # should be 9254 rows but with more columns now
#print(df.head())

# === STEP 4: Pandas Operations ===

# 1. Filter to adults only
# Filtering to adults (18+) because NHANES includes children and we want health metrics comparable to ICU populations
adults = df[df['RIDAGEYR'] >= 18]
#print(adults.shape, "after filtering to adult")  # should be less than 9254

# 2. Select specific columns
adults[['RIDAGEYR', 'RIAGENDR', 'BMXBMI']]

# 3. Average BMI by gender
#print(adults.groupby('RIAGENDR')['BMXBMI'].mean())

# 4. Multiple stats at once
#print(adults.groupby('RIAGENDR')['BMXBMI'].agg(['mean', 'median', 'std']))

# 5. Count of each race/ethnicity category
#print(adults['RIDRETH3'].value_counts())

# 6. How many missing values per column
#print(adults.isna().sum())

# 7. Create a complete-cases subset (no missing BMI or BP)
complete = adults.dropna(subset=['BMXBMI', 'BPXSY1'])
#print(complete.shape)

# 8. Fill missing blood pressure with median
adults['BPXSY1_filled'] = adults['BPXSY1'].fillna(adults['BPXSY1'].median())

# 9. Summary stats for all numeric columns
#print(adults.describe())

# 10. Average systolic BP by age group (preview of step 7)
adults['age_group'] = pd.cut(adults['RIDAGEYR'], bins=[18,30,45,60,75,100],
                              labels=['18-29','30-44','45-59','60-74','75+'])
#print(adults.groupby('age_group')['BPXSY1'].mean().sort_values())

# === STEP 5: data cleaning ===
# See the full missing picture

missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(1)
print("====== MISSING VALUES (%) ======")
print(missing_pct[missing_pct > 0])




#=========Step 6: Visualisations ==============

import matplotlib.pyplot as plt

# 1. Histogram — BMI distribution
#distribution of BMI across the year 2017-2018
#Highest BMI range is in the 25-30 bin
#steep positive skew 30 onwards
bins = range(10, 100, 5)
df['BMXBMI'].hist(bins=bins, edgecolor='black')
plt.xlabel('BMI')
plt.ylabel('Count')
plt.title('Distribution of BMI in NHANES 2017-2018')
plt.xticks(bins)
plt.grid(bins)
plt.show()

# 2. Bar chart — Average BMI by gender
#female have slightly higher mean of BMI
df.groupby(adults['RIAGENDR'])['BMXBMI'].mean().plot(kind='bar')
plt.xticks([0, 1], ['Male', 'Female'], rotation=0)
plt.ylabel('Mean BMI')
plt.xlabel('Gender')
plt.title('Average BMI by Gender')
plt.show()

# 3. Scatter plot — Age vs Systolic BP
# Observation: systolic BP trends upward with age, especially after 50. Expect this to be a strong predictor.
plt.scatter(adults['RIDAGEYR'], adults['BPXSY1'], alpha=0.1, s=5)
plt.xlabel('Age (years)')
plt.ylabel('Systolic Blood Pressure')
plt.title('Age vs Systolic Blood Pressure')
plt.xticks(range(18, 85, 5))
plt.show()

#=======creating bins for age groups========

# Create age bins
df['age_group'] = pd.cut(adults['RIDAGEYR'], bins=[18, 30, 45, 60, 75, 100],
                          labels=['18-29', '30-44', '45-59', '60-74', '75+'])

# The answer
df.groupby(['age_group', 'RIAGENDR'])['BMXBMI'].mean().unstack()

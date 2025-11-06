"""
Titanic Survival Analysis
Exploratory Data Analysis on Titanic Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the dataset
print("Loading Titanic dataset...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print("\n" + "="*80)

# ============================================================================
# 1. DATA EXPLORATION AND UNDERSTANDING
# ============================================================================

print("\n1. DATA EXPLORATION")
print("="*80)

print("\nFirst few rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_percent
})
print(missing_df[missing_df['Missing Count'] > 0])

# ============================================================================
# 2. DATA CLEANING AND PREPARATION
# ============================================================================

print("\n\n2. DATA CLEANING")
print("="*80)

# Create a copy for cleaning
df_clean = df.copy()

# Handle missing Age values - fill with median by Pclass
df_clean['Age'].fillna(df_clean.groupby('Pclass')['Age'].transform('median'), inplace=True)

# Handle missing Embarked values - fill with mode
df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)

# Drop Cabin column due to too many missing values
df_clean.drop('Cabin', axis=1, inplace=True)

# Create new features
df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)

# Create Age groups
df_clean['AgeGroup'] = pd.cut(df_clean['Age'], 
                               bins=[0, 12, 18, 35, 60, 100],
                               labels=['Child', 'Teenager', 'Adult', 'Middle-Aged', 'Senior'])

# Create Fare groups
df_clean['FareGroup'] = pd.qcut(df_clean['Fare'], q=4, 
                                 labels=['Low', 'Medium', 'High', 'Very High'],
                                 duplicates='drop')

print("Data cleaning completed!")
print(f"Cleaned dataset shape: {df_clean.shape}")

# ============================================================================
# 3. SURVIVAL ANALYSIS BY DIFFERENT FACTORS
# ============================================================================

print("\n\n3. SURVIVAL ANALYSIS")
print("="*80)

# Overall survival rate
overall_survival = df_clean['Survived'].mean() * 100
print(f"\nOverall Survival Rate: {overall_survival:.2f}%")

# Survival by Gender
print("\nSurvival by Gender:")
gender_survival = df_clean.groupby('Sex')['Survived'].agg(['mean', 'count'])
gender_survival['mean'] = gender_survival['mean'] * 100
gender_survival.columns = ['Survival Rate (%)', 'Count']
print(gender_survival)

# Survival by Class
print("\nSurvival by Passenger Class:")
class_survival = df_clean.groupby('Pclass')['Survived'].agg(['mean', 'count'])
class_survival['mean'] = class_survival['mean'] * 100
class_survival.columns = ['Survival Rate (%)', 'Count']
print(class_survival)

# Survival by Age Group
print("\nSurvival by Age Group:")
age_survival = df_clean.groupby('AgeGroup')['Survived'].agg(['mean', 'count'])
age_survival['mean'] = age_survival['mean'] * 100
age_survival.columns = ['Survival Rate (%)', 'Count']
print(age_survival)

# Survival by Family Size
print("\nSurvival by Family Size:")
family_survival = df_clean.groupby('FamilySize')['Survived'].agg(['mean', 'count'])
family_survival['mean'] = family_survival['mean'] * 100
family_survival.columns = ['Survival Rate (%)', 'Count']
print(family_survival)

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================

print("\n\n4. CREATING VISUALIZATIONS")
print("="*80)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Overall Survival Distribution
plt.subplot(3, 3, 1)
survival_counts = df_clean['Survived'].value_counts()
plt.pie(survival_counts, labels=['Did Not Survive', 'Survived'], 
        autopct='%1.1f%%', colors=['#ff6b6b', '#51cf66'], startangle=90)
plt.title('Overall Survival Distribution', fontsize=14, fontweight='bold')

# 2. Survival by Gender
plt.subplot(3, 3, 2)
sns.barplot(data=df_clean, x='Sex', y='Survived', palette='Set2')
plt.title('Survival Rate by Gender', fontsize=14, fontweight='bold')
plt.ylabel('Survival Rate')
plt.xlabel('Gender')

# 3. Survival by Passenger Class
plt.subplot(3, 3, 3)
sns.barplot(data=df_clean, x='Pclass', y='Survived', palette='viridis')
plt.title('Survival Rate by Passenger Class', fontsize=14, fontweight='bold')
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class')

# 4. Survival by Gender and Class
plt.subplot(3, 3, 4)
sns.barplot(data=df_clean, x='Pclass', y='Survived', hue='Sex', palette='coolwarm')
plt.title('Survival Rate by Class and Gender', fontsize=14, fontweight='bold')
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class')
plt.legend(title='Gender')

# 5. Age Distribution
plt.subplot(3, 3, 5)
plt.hist([df_clean[df_clean['Survived']==1]['Age'], 
          df_clean[df_clean['Survived']==0]['Age']], 
         bins=30, label=['Survived', 'Did Not Survive'], 
         color=['#51cf66', '#ff6b6b'], alpha=0.7)
plt.title('Age Distribution by Survival', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()

# 6. Survival by Age Group
plt.subplot(3, 3, 6)
sns.barplot(data=df_clean, x='AgeGroup', y='Survived', palette='rocket')
plt.title('Survival Rate by Age Group', fontsize=14, fontweight='bold')
plt.ylabel('Survival Rate')
plt.xlabel('Age Group')
plt.xticks(rotation=45)

# 7. Survival by Family Size
plt.subplot(3, 3, 7)
sns.barplot(data=df_clean, x='FamilySize', y='Survived', palette='mako')
plt.title('Survival Rate by Family Size', fontsize=14, fontweight='bold')
plt.ylabel('Survival Rate')
plt.xlabel('Family Size')

# 8. Fare Distribution by Survival
plt.subplot(3, 3, 8)
plt.hist([df_clean[df_clean['Survived']==1]['Fare'], 
          df_clean[df_clean['Survived']==0]['Fare']], 
         bins=30, label=['Survived', 'Did Not Survive'], 
         color=['#51cf66', '#ff6b6b'], alpha=0.7)
plt.title('Fare Distribution by Survival', fontsize=14, fontweight='bold')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(0, 300)

# 9. Correlation Heatmap
plt.subplot(3, 3, 9)
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
correlation = df_clean[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('titanic_analysis_visualizations.png', dpi=300, bbox_inches='tight')
print("Visualizations saved as 'titanic_analysis_visualizations.png'")
plt.show()

# ============================================================================
# 5. KEY INSIGHTS AND FINDINGS
# ============================================================================

print("\n\n5. KEY INSIGHTS")
print("="*80)

print("""
KEY FINDINGS FROM TITANIC SURVIVAL ANALYSIS:

1. GENDER IMPACT:
   - Women had a significantly higher survival rate than men
   - This reflects the "women and children first" policy

2. PASSENGER CLASS:
   - First-class passengers had the highest survival rate
   - Third-class passengers had the lowest survival rate
   - Socioeconomic status played a major role in survival

3. AGE FACTOR:
   - Children had higher survival rates
   - Middle-aged passengers had lower survival rates
   - The elderly faced challenges in survival

4. FAMILY SIZE:
   - Passengers with small families (2-4 members) had better survival rates
   - Solo travelers and very large families had lower survival rates
   - Having some family support helped, but too many dependents was a disadvantage

5. FARE CORRELATION:
   - Higher fare prices correlated with better survival rates
   - This is linked to passenger class and cabin location

6. COMBINED FACTORS:
   - First-class women had the highest survival rate
   - Third-class men had the lowest survival rate
   - Multiple factors interacted to determine survival outcomes
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

# Save cleaned dataset
df_clean.to_csv('titanic_cleaned.csv', index=False)
print("\nCleaned dataset saved as 'titanic_cleaned.csv'")

import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

cgpa = np.round(np.random.uniform(5.0, 10.0, n), 2)
internships = np.random.randint(0, 4, n)
projects = np.random.randint(0, 5, n)
aptitude_score = np.random.randint(40, 100, n)
soft_skills = np.random.randint(1, 6, n)  # 1-5 rating
backlog = np.random.randint(0, 3, n)

# Placement logic (realistic)
score = (
    (cgpa - 5) / 5 * 40 +
    internships * 10 +
    projects * 5 +
    (aptitude_score - 40) / 60 * 20 +
    soft_skills * 3 -
    backlog * 15
)

placed = (score + np.random.normal(0, 8, n) > 35).astype(int)

df = pd.DataFrame({
    'CGPA': cgpa,
    'Internships': internships,
    'Projects': projects,
    'Aptitude_Score': aptitude_score,
    'Soft_Skills': soft_skills,
    'Backlogs': backlog,
    'Placed': placed
})

df.to_csv('data/placement_data.csv', index=False)
print(f"Dataset created: {len(df)} rows")
print(f"Placement rate: {df['Placed'].mean()*100:.1f}%")
print(df.head())

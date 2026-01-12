import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the results
df = pd.read_csv('lhi_results_exact_only.csv')

# 2. Pre-process the scores
# Convert LHI_Score to numeric, turning 'N/A' into NaN, then drop them
df['LHI_Score'] = pd.to_numeric(df['LHI_Score'], errors='coerce')
scores = df['LHI_Score'].dropna()

# 3. Create the Plot
plt.figure(figsize=(10, 6))
plt.hist(scores, bins=20, color='skyblue', edgecolor='black', alpha=0.8)

# 4. Add Labels and Formatting
plt.title('Distribution of Legal Hallucination Index (LHI) Scores', fontsize=14)
plt.xlabel('LHI Score (1.0 = No Hallucinations)', fontsize=12)
plt.ylabel('Number of Documents', fontsize=12)
plt.axvline(scores.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {scores.mean():.2f}')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

# 5. Save and Show
plt.savefig('lhi_histogram_exact_match.png')
print("Histogram successfully saved as 'lhi_histogram.png'")
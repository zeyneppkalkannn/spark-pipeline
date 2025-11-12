import pandas as pd
import matplotlib.pyplot as plt

results = {
    'Scenario': ['No Drift', 'Low Scale Drift', 'Medium Scale Drift', 
                 'High Scale Drift', 'Mean Shift', 'Added Noise', '20% Data Deletion'],
    'dP': [0.00, 0.45, 45.80, 38.26, 0.00, 27.57, 1.32],
    'dE_PCA': [0.09, 0.52, 0.31, 1.41, 14.70, 12.68, 0.08]
}

df = pd.DataFrame(results)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(df['Scenario'], df['dP'], color='#3498db', alpha=0.7)
axes[0].axvline(x=30, color='orange', linestyle='--', alpha=0.5)
axes[0].axvline(x=50, color='red', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Drift Value (0-100)')
axes[0].set_title('dP Metric')
axes[0].grid(axis='x', alpha=0.3)

axes[1].barh(df['Scenario'], df['dE_PCA'], color='#e74c3c', alpha=0.7)
axes[1].axvline(x=30, color='orange', linestyle='--', alpha=0.5)
axes[1].axvline(x=50, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Drift Value (0-100)')
axes[1].set_title('dE_PCA Metric (RECOMMENDED)')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('drift_comparison.png', dpi=300, bbox_inches='tight')
print("Gorsel kaydedildi: drift_comparison.png")
print(df)
"""
DATA DRIFT DETECTION - DEMO SCRIPT
===================================

This script demonstrates how to use the Data Drift Detector
with synthetic time series data.

Based on González-Cebrián et al. (2024) methodology.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_drift_detector import DataDriftDetector
import warnings
warnings.filterwarnings('ignore')


def generate_time_series_data(n_samples=1000, n_features=10, 
                              trend=False, seasonality=False, 
                              noise_level=0.1, seed=42):
    """
    Generate synthetic time series data
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        trend: Add trend component
        seasonality: Add seasonal component
        noise_level: Noise standard deviation
        seed: Random seed
        
    Returns:
        DataFrame with time series data
    """
    np.random.seed(seed)
    
    
    t = np.arange(n_samples)
    
    
    data = np.zeros((n_samples, n_features))
    
    for i in range(n_features):
    
        signal = np.random.randn(n_samples) * noise_level
        
        
        if trend:
            signal += (t / n_samples) * np.random.uniform(0.5, 2.0)
        
        
        if seasonality:
            period = np.random.uniform(50, 200)
            signal += np.sin(2 * np.pi * t / period) * np.random.uniform(0.5, 1.5)
        
        data[:, i] = signal
    
    
    columns = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    
    return df


def simulate_drift_scenarios(reference_data, drift_type='scale', drift_intensity=0.5):
    """
    Simulate different types of data drift
    
    Args:
        reference_data: Reference DataFrame
        drift_type: Type of drift ('scale', 'shift', 'noise', 'missing')
        drift_intensity: Intensity of drift (0.0 to 1.0)
        
    Returns:
        Drifted DataFrame
    """
    current_data = reference_data.copy()
    n_features = current_data.shape[1]
    n_affected = max(1, int(n_features * drift_intensity))
    affected_cols = np.random.choice(current_data.columns, n_affected, replace=False)
    
    if drift_type == 'scale':
        
        for col in affected_cols:
            current_data[col] = np.sign(current_data[col]) * np.abs(current_data[col]) ** (1/3)
        print(f"Applied SCALE drift to {n_affected}/{n_features} features")
    
    elif drift_type == 'shift':

        for col in affected_cols:
            shift = current_data[col].std() * 2
            current_data[col] += shift
        print(f"Applied SHIFT drift to {n_affected}/{n_features} features")
    
    elif drift_type == 'noise':
        # Add noise
        for col in affected_cols:
            noise = np.random.randn(len(current_data)) * current_data[col].std()
            current_data[col] += noise
        print(f"Applied NOISE drift to {n_affected}/{n_features} features")
    
    elif drift_type == 'missing':
        # Remove random rows (deletion scenario)
        keep_ratio = 1.0 - drift_intensity
        n_keep = int(len(current_data) * keep_ratio)
        current_data = current_data.sample(n=n_keep, random_state=42).reset_index(drop=True)
        print(f"DELETED {100*drift_intensity:.0f}% of records")
    
    return current_data


def run_demo():
    """Run the complete demo"""
    
    print("\n" + "="*80)
    print("DATA DRIFT DETECTION - COMPREHENSIVE DEMO")
    print("="*80)
    
    # 1. GENERATE DATA
    print("\n[STEP 1] Generating Synthetic Time Series Data")
    print("-" * 80)
    
    # Reference data (Primary Source)
    reference_data = generate_time_series_data(
        n_samples=500,
        n_features=8,
        trend=True,
        seasonality=True,
        noise_level=0.2,
        seed=42
    )
    print(f"✓ Reference data shape: {reference_data.shape}")
    print(f"✓ Features: {list(reference_data.columns)}")
    
    # 2. FIT DETECTOR
    print("\n[STEP 2] Fitting Data Drift Detector")
    print("-" * 80)
    
    detector = DataDriftDetector(n_components_pca=0.90, random_state=42)
    detector.fit(reference_data, build_ae=True, ae_params={
        'hidden_layers': [32, 16],
        'activation': 'relu',
        'epochs': 20,
        'batch_size': 32,
        'noise_factor': 0.1
    })
    
    # 3. TEST SCENARIOS
    print("\n[STEP 3] Testing Different Drift Scenarios")
    print("-" * 80)
    
    scenarios = [
        ('No Drift', 'scale', 0.0),
        ('Low Scale Drift', 'scale', 0.2),
        ('Medium Scale Drift', 'scale', 0.4),
        ('High Scale Drift', 'scale', 0.6),
        ('Mean Shift', 'shift', 0.3),
        ('Added Noise', 'noise', 0.5),
        ('20% Data Deletion', 'missing', 0.2),
    ]
    
    results_summary = []
    
    for scenario_name, drift_type, intensity in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*80}")
        
        # Generate drifted data
        if drift_type == 'scale' and intensity == 0.0:
            # No drift - use reference data
            current_data = reference_data.copy()
            print("Using reference data (no changes)")
        else:
            current_data = simulate_drift_scenarios(reference_data, drift_type, intensity)
        
        # Calculate drift metrics
        results = detector.calculate_all_metrics(current_data)
        
        # Print results
        detector.print_results(results)
        
        # Store for summary
        results_summary.append({
            'Scenario': scenario_name,
            'dP': results['dP'],
            'dE_PCA': results['dE_PCA'],
            'dE_AE': results['dE_AE'] if results['dE_AE'] is not None else np.nan
        })
    
    # 4. SUMMARY TABLE
    print("\n" + "="*80)
    print("SUMMARY OF ALL SCENARIOS")
    print("="*80 + "\n")
    
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    
    # 5. VISUALIZATION
    print("\n[STEP 4] Generating Visualization")
    print("-" * 80)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['dP', 'dE_PCA', 'dE_AE']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx]
        
        values = summary_df[metric].values
        scenarios = summary_df['Scenario'].values
        
        bars = ax.barh(scenarios, values, color=color, alpha=0.7)
        
        # Add threshold lines
        ax.axvline(x=30, color='orange', linestyle='--', alpha=0.5, label='Threshold: 30')
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Threshold: 50')
        
        ax.set_xlabel('Drift Value (0-100)', fontsize=10)
        ax.set_title(f'{metric} Metric', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            if not np.isnan(width):
                ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                       f'{width:.1f}', 
                       va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('drift_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved: drift_comparison.png")
    
    # 6. INTERPRETATION GUIDE
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("""
Metric Interpretations:
----------------------

dP (PCA Loadings):
  • 0-30:   Low drift - correlation structure preserved
  • 30-50:  Moderate drift - some structural changes
  • 50-100: High drift - major structural differences

dE,PCA (PCA Reconstruction Error):
  • 0-30:   Low drift - data patterns similar
  • 30-50:  Moderate drift - noticeable differences
  • 50-100: High drift - data very different (>50% like random noise)

dE,AE (Autoencoder Reconstruction Error):
  • 0-30:   Low drift - nonlinear patterns preserved
  • 30-50:  Moderate drift - pattern changes detected
  • 50-100: High drift - substantial nonlinear changes

Recommendations:
---------------
• dE,PCA < 30:   Minor version update (e.g., 1.1.0 → 1.2.0)
• dE,PCA 30-50:  Review and decide on minor or major update
• dE,PCA > 50:   Major version update (e.g., 1.x.0 → 2.0.0)

Based on González-Cebrián et al. (2024):
"Standardised Versioning of Datasets: a FAIR–compliant Proposal"
Scientific Data, Nature Publishing Group
""")
    
    print("\n" + "="*80)
    print("✓ DEMO COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run the demo
    run_demo()
    
    print("\n📊 Output file created:")
    print("   → /mnt/user-data/outputs/drift_comparison.png")

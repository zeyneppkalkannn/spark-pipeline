"""
REAL DATA EXAMPLE - Data Drift Detection with CSV Files
========================================================

This script shows how to use the Data Drift Detector with real CSV data.
Perfect for your assignment!
"""

import numpy as np
import pandas as pd
from data_drift_detector import DataDriftDetector
import warnings
warnings.filterwarnings('ignore')


def example_with_csv():
    """
    Example: Using Data Drift Detector with CSV files
    
    This demonstrates the typical workflow:
    1. Load reference data (training/historical data)
    2. Fit the detector
    3. Load current data (new/production data)
    4. Calculate drift metrics
    5. Interpret results
    """
    
    print("\n" + "="*80)
    print("DATA DRIFT DETECTION - CSV FILE EXAMPLE")
    print("="*80)
    
    # =========================================================================
    # STEP 1: LOAD YOUR DATA
    # =========================================================================
    print("\n[STEP 1] Loading Data")
    print("-" * 80)
    
    # Replace these with your actual CSV file paths
    # reference_data = pd.read_csv('path/to/your/reference_data.csv')
    # current_data = pd.read_csv('path/to/your/current_data.csv')
    
    # For this example, we'll generate synthetic data
    print("NOTE: Replace this with your actual CSV file paths!")
    print("Example:")
    print("  reference_data = pd.read_csv('training_data_2023.csv')")
    print("  current_data = pd.read_csv('production_data_2024.csv')")
    
    # Generate example data
    np.random.seed(42)
    
    # Reference data (e.g., data from 2023)
    reference_data = pd.DataFrame({
        'temperature': np.random.normal(20, 5, 1000),
        'humidity': np.random.normal(60, 15, 1000),
        'pressure': np.random.normal(1013, 10, 1000),
        'wind_speed': np.random.gamma(2, 2, 1000),
        'rainfall': np.random.exponential(2, 1000)
    })
    
    # Current data (e.g., data from 2024 with some drift)
    current_data = pd.DataFrame({
        'temperature': np.random.normal(22, 5, 500),  # Mean shifted
        'humidity': np.random.normal(65, 18, 500),    # Mean + variance changed
        'pressure': np.random.normal(1013, 10, 500),  # No change
        'wind_speed': np.random.gamma(2.5, 2, 500),   # Distribution changed
        'rainfall': np.random.exponential(2, 500)     # No change
    })
    
    print(f"\n✓ Reference data loaded: {reference_data.shape}")
    print(f"  Columns: {list(reference_data.columns)}")
    print(f"\n✓ Current data loaded: {current_data.shape}")
    
    # =========================================================================
    # STEP 2: FIT THE DETECTOR
    # =========================================================================
    print("\n[STEP 2] Fitting Data Drift Detector")
    print("-" * 80)
    
    detector = DataDriftDetector(
        n_components_pca=0.90,  # Explain 90% of variance
        random_state=42
    )
    
    # Fit on reference data
    # Note: Set build_ae=False if TensorFlow is not installed
    detector.fit(reference_data, build_ae=False)
    
    # =========================================================================
    # STEP 3: DETECT DRIFT
    # =========================================================================
    print("\n[STEP 3] Calculating Drift Metrics")
    print("-" * 80)
    
    # Calculate all metrics
    results = detector.calculate_all_metrics(current_data)
    
    # Print results
    detector.print_results(results)
    
    # =========================================================================
    # STEP 4: DETAILED ANALYSIS
    # =========================================================================
    print("\n[STEP 4] Detailed Analysis")
    print("-" * 80)
    
    print("\nData Statistics Comparison:")
    print("-" * 40)
    
    # Compare means
    print("\nMean Values:")
    comparison = pd.DataFrame({
        'Reference': reference_data.mean(),
        'Current': current_data.mean(),
        'Change %': ((current_data.mean() - reference_data.mean()) / 
                     reference_data.mean() * 100)
    })
    print(comparison.round(2))
    
    # Compare standard deviations
    print("\nStandard Deviation:")
    comparison_std = pd.DataFrame({
        'Reference': reference_data.std(),
        'Current': current_data.std(),
        'Change %': ((current_data.std() - reference_data.std()) / 
                     reference_data.std() * 100)
    })
    print(comparison_std.round(2))
    
    # =========================================================================
    # STEP 5: RECOMMENDATIONS
    # =========================================================================
    print("\n[STEP 5] Recommendations")
    print("-" * 80)
    
    dP = results['dP']
    dE_PCA = results['dE_PCA']
    
    print("\nBased on González-Cebrián et al. (2024) methodology:")
    print("-" * 50)
    
    if dE_PCA < 10:
        print("✓ VERY LOW DRIFT")
        print("  → No action needed")
        print("  → Data patterns are highly similar")
        
    elif dE_PCA < 30:
        print("⚠ LOW TO MODERATE DRIFT")
        print("  → Consider minor version update (e.g., v1.1 → v1.2)")
        print("  → Monitor trends")
        print("  → Document changes")
        
    elif dE_PCA < 50:
        print("⚠⚠ MODERATE DRIFT")
        print("  → Review data quality")
        print("  → Consider retraining models")
        print("  → Update documentation")
        
    else:
        print("🔴 HIGH DRIFT DETECTED!")
        print("  → MAJOR VERSION UPDATE NEEDED (e.g., v1.x → v2.0)")
        print("  → Investigate root causes")
        print("  → Retrain all models")
        print("  → Update data pipelines")
    
    # Feature-level insights
    print("\n\nFeature-Level Insights:")
    print("-" * 50)
    
    for col in reference_data.columns:
        ref_mean = reference_data[col].mean()
        cur_mean = current_data[col].mean()
        change_pct = ((cur_mean - ref_mean) / ref_mean * 100)
        
        if abs(change_pct) > 10:
            print(f"⚠ {col:15s}: {change_pct:+6.1f}% change in mean")
        else:
            print(f"✓ {col:15s}: {change_pct:+6.1f}% change in mean")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80 + "\n")
    
    return detector, results


def batch_analysis_example():
    """
    Example: Analyzing multiple batches of data over time
    
    This is useful for monitoring drift in production systems
    """
    
    print("\n" + "="*80)
    print("BATCH ANALYSIS EXAMPLE - Monitoring Drift Over Time")
    print("="*80)
    
    np.random.seed(42)
    
    # Generate reference data
    reference_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(0, 1, 1000),
        'feature_3': np.random.normal(0, 1, 1000),
    })
    
    # Fit detector
    detector = DataDriftDetector()
    detector.fit(reference_data, build_ae=False)
    
    # Simulate 6 monthly batches with increasing drift
    batches = []
    for month in range(1, 7):
        # Gradually increasing drift
        drift_factor = month * 0.1
        
        batch = pd.DataFrame({
            'feature_1': np.random.normal(drift_factor, 1, 200),
            'feature_2': np.random.normal(drift_factor, 1, 200),
            'feature_3': np.random.normal(0, 1, 200),
        })
        
        results = detector.calculate_all_metrics(batch)
        
        batches.append({
            'Month': f'Month {month}',
            'dP': results['dP'],
            'dE_PCA': results['dE_PCA'],
            'Drift_Factor': drift_factor
        })
    
    # Summary
    batch_df = pd.DataFrame(batches)
    
    print("\nDrift Over Time:")
    print("-" * 60)
    print(batch_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("✓ You can see how drift increases over time!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run CSV example
    detector, results = example_with_csv()
    
    print("\n" + "─"*80)
    
    # Run batch analysis example
    batch_analysis_example()
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                          IMPORTANT NOTES                                   ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  1. Replace the example data with your actual CSV files:                  ║
║     reference_data = pd.read_csv('your_reference_data.csv')              ║
║     current_data = pd.read_csv('your_current_data.csv')                  ║
║                                                                            ║
║  2. Make sure your data is numeric (no text columns)                      ║
║                                                                            ║
║  3. The three metrics complement each other:                              ║
║     • dP: Detects changes in correlation structure                        ║
║     • dE,PCA: Detects overall distribution changes (RECOMMENDED)          ║
║     • dE,AE: Detects complex nonlinear changes (needs TensorFlow)         ║
║                                                                            ║
║  4. Focus on dE,PCA for decision making (best trade-off)                  ║
║                                                                            ║
║  5. For your Thursday deadline, you can:                                  ║
║     a) Use synthetic data (like this example)                             ║
║     b) Use real CSV data from your project                                ║
║     c) Use datasets from the paper (see README.md)                        ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

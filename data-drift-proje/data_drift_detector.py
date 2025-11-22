
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import UnivariateSpline
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DataDriftDetector:
   
    
    def __init__(self, n_components_pca: float = 0.90, random_state: int = 42):
        
        self.n_components_pca = n_components_pca
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Models to be fitted
        self.pca_model = None
        self.ae_model = None
        
        # Reference data
        self.X_reference_scaled = None
        self.X_reference_raw = None
        
        # Splines for E-drift metrics
        self.splines_pca = []
        self.splines_ae = []
        
        # Results storage
        self.drift_history = []
    
    
    def fit(self, X_reference: pd.DataFrame, build_ae: bool = True, 
            ae_params: Optional[Dict] = None):
       
        print("=" * 70)
        print("FITTING DATA DRIFT DETECTOR")
        print("=" * 70)
        
        # Store reference data
        self.X_reference_raw = X_reference.copy()
        
        # 1. STANDARDIZE DATA
        print("\n[1/4] Standardizing data (mean=0, std=1)...")
        self.X_reference_scaled = self.scaler.fit_transform(X_reference)
        print(f"   ✓ Shape: {self.X_reference_scaled.shape}")
        print(f"   ✓ Features: {X_reference.shape[1]}")
        print(f"   ✓ Samples: {X_reference.shape[0]}")
        
        # 2. FIT PCA MODEL
        print("\n[2/4] Fitting PCA model...")
        self.pca_model = PCA(n_components=self.n_components_pca, 
                             random_state=self.random_state)
        self.pca_model.fit(self.X_reference_scaled)
        
        n_components = self.pca_model.n_components_
        var_explained = self.pca_model.explained_variance_ratio_.sum()
        print(f"   ✓ Components: {n_components}")
        print(f"   ✓ Variance explained: {var_explained:.2%}")
        
        # 3. BUILD SPLINES FOR dE,PCA
        print("\n[3/4] Building splines for dE,PCA metric...")
        self._build_splines_pca(n_repetitions=10)
        print(f"   ✓ Built {len(self.splines_pca)} splines")
        
        # 4. FIT AUTOENCODER (if requested)
        if build_ae:
            print("\n[4/4] Building Autoencoder model...")
            self._build_autoencoder(ae_params)
            print("\n[4/4] Building splines for dE,AE metric...")
            self._build_splines_ae(n_repetitions=10)
            print(f"   ✓ Built {len(self.splines_ae)} splines")
        else:
            print("\n[4/4] Skipping Autoencoder (build_ae=False)")
        
        print("\n" + "=" * 70)
        print("✓ FITTING COMPLETE!")
        print("=" * 70)
        
        return self
    
    
    def _build_splines_pca(self, n_repetitions: int = 10):
        """
        Build splines for dE,PCA metric by permuting reference data
        
        Args:
            n_repetitions: Number of repetitions for each permutation level
        """
        permutation_levels = np.arange(0.01, 1.01, 0.1)  # 1% to 100%
        
        for rep in range(n_repetitions):
            mse_values = []
            perm_levels = []
            
            for perm_pct in permutation_levels:
                # Permute percentage of cells
                X_permuted = self._permute_data(self.X_reference_scaled, perm_pct)
                
                # Reconstruct with PCA
                X_reconstructed = self.pca_model.inverse_transform(
                    self.pca_model.transform(X_permuted)
                )
                
                # Calculate MSE
                mse = np.mean((self.X_reference_scaled - X_reconstructed) ** 2)
                
                mse_values.append(mse)
                perm_levels.append(perm_pct * 100)
            
            # Fit spline: MSE → percentage of permuted values
            spline = UnivariateSpline(mse_values, perm_levels, s=0, k=3)
            self.splines_pca.append(spline)
    
    
    def _build_splines_ae(self, n_repetitions: int = 10):
        
        if self.ae_model is None:
            return
        
        permutation_levels = np.arange(0.01, 1.01, 0.1)
        
        for rep in range(n_repetitions):
            mse_values = []
            perm_levels = []
            
            for perm_pct in permutation_levels:
                X_permuted = self._permute_data(self.X_reference_scaled, perm_pct)
                
                # Reconstruct with Autoencoder
                X_reconstructed = self.ae_model.predict(X_permuted, verbose=0)
                
                # Calculate MSE
                mse = np.mean((self.X_reference_scaled - X_reconstructed) ** 2)
                
                mse_values.append(mse)
                perm_levels.append(perm_pct * 100)
            
            # Fit spline
            spline = UnivariateSpline(mse_values, perm_levels, s=0, k=3)
            self.splines_ae.append(spline)
    
    
    def _permute_data(self, X: np.ndarray, perm_pct: float) -> np.ndarray:
        """
        Randomly permute a percentage of cells in the data
        
        Args:
            X: Input data
            perm_pct: Percentage of cells to permute (0.0 to 1.0)
            
        Returns:
            Permuted data
        """
        X_perm = X.copy()
        n_cells = X.size
        n_to_permute = int(n_cells * perm_pct)
        
        # Random indices to permute
        flat_indices = np.random.choice(n_cells, n_to_permute, replace=False)
        
        # Flatten, permute, reshape
        X_flat = X_perm.flatten()
        X_flat[flat_indices] = np.random.permutation(X_flat[flat_indices])
        X_perm = X_flat.reshape(X.shape)
        
        return X_perm
    
    
    def _build_autoencoder(self, ae_params: Optional[Dict] = None):
        
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            print("   ⚠ TensorFlow not installed. Skipping Autoencoder.")
            return
        
        # Default parameters
        if ae_params is None:
            ae_params = {
                'hidden_layers': [64, 32],
                'activation': 'relu',
                'epochs': 20,
                'batch_size': 32,
                'noise_factor': 0.1
            }
        
        input_dim = self.X_reference_scaled.shape[1]
        
        # Build encoder
        encoder_input = layers.Input(shape=(input_dim,))
        x = encoder_input
        
        for units in ae_params['hidden_layers']:
            x = layers.Dense(units, activation=ae_params['activation'])(x)
        
        # Latent representation
        latent_dim = ae_params['hidden_layers'][-1]
        
        # Build decoder
        for units in reversed(ae_params['hidden_layers'][:-1]):
            x = layers.Dense(units, activation=ae_params['activation'])(x)
        
        decoder_output = layers.Dense(input_dim, activation='linear')(x)
        
        # Complete autoencoder
        self.ae_model = keras.Model(encoder_input, decoder_output)
        self.ae_model.compile(optimizer='adam', loss='mse')
        
        # Add noise for denoising
        noise_factor = ae_params['noise_factor']
        X_noisy = self.X_reference_scaled + noise_factor * np.random.normal(
            size=self.X_reference_scaled.shape
        )
        
        # Train
        self.ae_model.fit(
            X_noisy, 
            self.X_reference_scaled,
            epochs=ae_params['epochs'],
            batch_size=ae_params['batch_size'],
            verbose=0
        )
        
        print(f"   ✓ Architecture: {input_dim} → {' → '.join(map(str, ae_params['hidden_layers']))} → {input_dim}")
        print(f"   ✓ Epochs: {ae_params['epochs']}")
    
    
    def calculate_dp_drift(self, X_current: pd.DataFrame) -> float:
        """
        Calculate dP (PCA loadings-based) drift metric
        
        dP = 100 * (1 - Σ(λₐ |cos(pₐ⁽ᴾˢ⁾, pₐ⁽ᴿ⁾)|) / Σλₐ)
        
        Args:
            X_current: Current dataset (Revision)
            
        Returns:
            dP value (0-100)
        """
        # Standardize current data
        X_current_scaled = self.scaler.transform(X_current)
        
        # Fit new PCA on current data
        pca_current = PCA(n_components=self.n_components_pca, 
                          random_state=self.random_state)
        pca_current.fit(X_current_scaled)
        
        # Get loadings (components)
        P_ref = self.pca_model.components_  # shape: (n_components, n_features)
        P_cur = pca_current.components_
        
        # Get eigenvalues (explained variance)
        lambda_ref = self.pca_model.explained_variance_
        
        # Calculate cosine similarity for each component
        n_components = min(len(P_ref), len(P_cur))
        weighted_cos_sum = 0.0
        lambda_sum = 0.0
        
        for a in range(n_components):
            # Cosine of angle between loading vectors
            cos_sim = np.abs(np.dot(P_ref[a], P_cur[a]) / 
                            (np.linalg.norm(P_ref[a]) * np.linalg.norm(P_cur[a])))
            
            weighted_cos_sum += lambda_ref[a] * cos_sim
            lambda_sum += lambda_ref[a]
        
        # dP formula
        dp = 100 * (1 - weighted_cos_sum / lambda_sum)
        
        return float(dp)
    
    
    def calculate_de_pca_drift(self, X_current: pd.DataFrame) -> float:
        """
        Calculate dE,PCA (PCA reconstruction error with splines) drift metric
        
        Args:
            X_current: Current dataset (Revision)
            
        Returns:
            dE,PCA value (0-100)
        """
        # Standardize current data
        X_current_scaled = self.scaler.transform(X_current)
        
        # Reconstruct with reference PCA model
        X_reconstructed = self.pca_model.inverse_transform(
            self.pca_model.transform(X_current_scaled)
        )
        
        # Calculate MSE (between current data and its reconstruction)
        mse = np.mean((X_current_scaled - X_reconstructed) ** 2)
        
        # Use splines to estimate drift percentage
        drift_values = []
        for spline in self.splines_pca:
            try:
                drift = float(spline(mse))
                drift = np.clip(drift, 0, 100)  # Constrain to [0, 100]
                drift_values.append(drift)
            except:
                continue
        
        # Average across splines
        if len(drift_values) > 0:
            de_pca = np.mean(drift_values)
        else:
            de_pca = 0.0
        
        return float(de_pca)
    
    
    def calculate_de_ae_drift(self, X_current: pd.DataFrame) -> float:
        """
        Calculate dE,AE (Autoencoder reconstruction error with splines) drift metric
        
        Args:
            X_current: Current dataset (Revision)
            
        Returns:
            dE,AE value (0-100), or None if no autoencoder
        """
        if self.ae_model is None:
            return None
        
        # Standardize current data
        X_current_scaled = self.scaler.transform(X_current)
        
        # Reconstruct with autoencoder
        X_reconstructed = self.ae_model.predict(X_current_scaled, verbose=0)
        
        # Calculate MSE (between current data and its reconstruction)
        mse = np.mean((X_current_scaled - X_reconstructed) ** 2)
        
        # Use splines to estimate drift percentage
        drift_values = []
        for spline in self.splines_ae:
            try:
                drift = float(spline(mse))
                drift = np.clip(drift, 0, 100)
                drift_values.append(drift)
            except:
                continue
        
        # Average across splines
        if len(drift_values) > 0:
            de_ae = np.mean(drift_values)
        else:
            de_ae = 0.0
        
        return float(de_ae)
    
    
    def calculate_all_metrics(self, X_current: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate all drift metrics at once
        
        Args:
            X_current: Current dataset (Revision)
            
        Returns:
            Dictionary with all drift metrics
        """
        results = {
            'dP': self.calculate_dp_drift(X_current),
            'dE_PCA': self.calculate_de_pca_drift(X_current),
            'dE_AE': self.calculate_de_ae_drift(X_current)
        }
        
        # Store in history
        self.drift_history.append(results.copy())
        
        return results
    
    
    def get_interpretation(self, drift_value: float, metric_type: str) -> str:
        """
        Interpret drift value
        
        Args:
            drift_value: Drift value (0-100)
            metric_type: Type of metric ('dP', 'dE_PCA', 'dE_AE')
            
        Returns:
            Interpretation string
        """
        if metric_type == 'dP':
            if drift_value < 10:
                return "Very Low Drift - Covariance structure highly similar"
            elif drift_value < 30:
                return "Low Drift - Minor changes in covariance structure"
            elif drift_value < 50:
                return "Moderate Drift - Noticeable changes detected"
            elif drift_value < 70:
                return "High Drift - Significant structural changes"
            else:
                return "Very High Drift - Major version update recommended"
        
        else:  # dE_PCA or dE_AE
            if drift_value < 10:
                return "Very Low Drift - Data highly similar"
            elif drift_value < 30:
                return "Low Drift - Minor changes in distribution"
            elif drift_value < 50:
                return "Moderate Drift - Data still recognizable"
            elif drift_value < 70:
                return "High Drift - Substantial differences"
            else:
                return "Very High Drift - Data drastically different (like random noise)"
    
    
    def print_results(self, results: Dict[str, float]):
        """
        Print drift results in a formatted way
        
        Args:
            results: Dictionary with drift metrics
        """
        print("\n" + "=" * 70)
        print("DATA DRIFT DETECTION RESULTS")
        print("=" * 70)
        
        for metric, value in results.items():
            if value is not None:
                interp = self.get_interpretation(value, metric)
                print(f"\n{metric:12s}: {value:6.2f}  │  {interp}")
        
        print("\n" + "=" * 70)
        
        # Recommendation
        print("\nRECOMMENDATION:")
        de_pca = results.get('dE_PCA', 0)
        
        if de_pca < 30:
            print("✓ Minor version update (0.X.0)")
        elif de_pca < 70:
            print("⚠ Consider minor version update, review changes")
        else:
            print("⚡ Major version update recommended (X.0.0)")
        
        print("=" * 70 + "\n")


if __name__ == "__main__":
    print("Data Drift Detector Module")
    print("This module should be imported, not run directly.")
    print("\nExample usage:")
    print("  from data_drift_detector import DataDriftDetector")
    print("  detector = DataDriftDetector()")
    print("  detector.fit(reference_data)")
    print("  results = detector.calculate_all_metrics(current_data)")

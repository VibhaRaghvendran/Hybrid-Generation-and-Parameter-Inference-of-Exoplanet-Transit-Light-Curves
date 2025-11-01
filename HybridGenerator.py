#HYBRID KEPLER LIGHT CURVE GENERATOR

# CELL 1: All Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

print("=" * 70)
print("HYBRID KEPLER LIGHT CURVE GENERATOR")
print("=" * 70)
print("VAE learns realistic stellar variability")
print("Physics model adds accurate transits")
print("=" * 70)

# CELL 2: Physics-Based Transit Generator
class PhysicsBasedTransitGenerator:
    """Generate realistic transits using physics equations"""

    def __init__(self, length=2000):
        self.length = length

    def generate_transit(self, time, period, t0, depth, duration):
        """
        Generate a transit signal using box model

        Args:
            time: time array
            period: orbital period
            t0: time of first transit center
            depth: transit depth (fraction)
            duration: transit duration
        """
        phase = np.mod(time - t0, period)
        transit = np.ones_like(time)

        # Main transit
        in_transit = np.abs(phase - period/2) < duration/2
        transit[in_transit] = 1 - depth

        # Early transit
        in_transit_early = phase < duration/2
        transit[in_transit_early] = 1 - depth

        return transit

    def add_noise(self, flux, noise_level=0.001):
        """Add photometric noise"""
        return flux + np.random.normal(0, noise_level, len(flux))

print("Physics generator defined")

# CELL 3: Dataset Loader
class HybridDataset(Dataset):
    """Load Kepler light curves for hybrid training"""

    def __init__(self, data_folder, target_length=2000):
        self.curves = []
        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

        print(f"Found {len(csv_files)} CSV files")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Find time and flux columns
                time_col = [col for col in df.columns if 'time' in col.lower()][0]
                flux_col = [col for col in df.columns if 'flux' in col.lower()][0]

                time = df[time_col].values
                flux = df[flux_col].values

                # Remove NaN
                mask = ~(np.isnan(time) | np.isnan(flux))
                flux = flux[mask]

                if len(flux) >= 100:
                    # Downsample with anti-aliasing
                    if len(flux) > target_length:
                        window = len(flux) // target_length
                        if window > 1:
                            flux = np.convolve(flux, np.ones(window)/window, mode='same')
                        indices = np.linspace(0, len(flux)-1, target_length, dtype=int)
                        flux = flux[indices]

                    # Normalize
                    flux_norm = (flux - np.mean(flux)) / (np.std(flux) + 1e-8)
                    self.curves.append(flux_norm)
            except Exception as e:
                continue

        print(f"Loaded {len(self.curves)} light curves")

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.curves[idx])

print("Dataset class defined")

# CELL 4: Transit Removal (Extract Stellar Variability)
def remove_transits(flux, threshold=3):
    """
    Remove transit dips to isolate stellar variability

    Args:
        flux: flux array
        threshold: sigma threshold for detecting transits
    """
    flux_clean = flux.copy()

    # Robust statistics
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    sigma = 1.4826 * mad

    # Find deep dips (transits)
    outliers = flux < (median - threshold * sigma)

    # Interpolate over transits
    if np.any(outliers):
        indices = np.arange(len(flux))
        good_indices = indices[~outliers]
        good_flux = flux[~outliers]

        if len(good_indices) > 10:
            flux_clean[outliers] = np.interp(indices[outliers], good_indices, good_flux)

    return flux_clean

print("Transit removal defined")

# CELL 5: Stellar Variability VAE
class StellarVAE(nn.Module):
    """VAE to learn stellar variability patterns"""

    def __init__(self, input_dim=2000, latent_dim=64):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

        self.flat_dim = 128 * (input_dim // 8)
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
        )

        self.latent_dim = latent_dim
        self.output_dim = input_dim

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, -1)
        x = self.decoder(x)
        x = x.squeeze(1)
        if x.size(1) != self.output_dim:
            x = F.interpolate(x.unsqueeze(1), size=self.output_dim, mode='linear').squeeze(1)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        with torch.no_grad():
            return self.decode(z)

print("Stellar VAE defined")

# CELL 6 UPDATE: Add this improved version of HybridLightCurveGenerator
class HybridLightCurveGenerator:
    """Combines VAE stellar variability with physics-based transits"""

    def __init__(self, vae_model, device='cpu'):
        self.vae = vae_model
        self.device = device
        self.physics_gen = PhysicsBasedTransitGenerator(length=2000)

    def add_realistic_noise(self, curve, noise_level=0.3):
        """
        Add realistic Kepler-like noise to synthetic curves
        
        Combines:
        - White noise (photon noise)
        - Correlated noise (instrumental drift)
        """
        # White noise (photon noise)
        white_noise = np.random.normal(0, noise_level, len(curve))
        
        # Low-frequency correlated noise (instrumental)
        # Create smooth variations using a low-pass filter
        corr_noise = np.random.normal(0, noise_level * 0.5, len(curve))
        window_size = 50  # Smooth over ~50 points
        kernel = np.ones(window_size) / window_size
        corr_noise = np.convolve(corr_noise, kernel, mode='same')
        
        # Combine noises
        total_noise = white_noise + corr_noise
        
        return curve + total_noise

    def generate(self, num_samples=1, add_transits=True, num_transits=None, 
                 noise_level=0.3, add_realistic_noise=True):
        """
        Generate hybrid light curves

        Args:
            num_samples: number of curves to generate
            add_transits: whether to add transits
            num_transits: number of transits (None = random 2-4)
            noise_level: amount of noise to add (0.3 matches real Kepler)
            add_realistic_noise: whether to add Kepler-like noise
        """
        # Generate stellar variability from VAE
        stellar_var = self.vae.sample(num_samples, self.device).cpu().numpy()

        if not add_transits:
            # Add noise even without transits
            if add_realistic_noise:
                stellar_var = np.array([self.add_realistic_noise(curve, noise_level) 
                                       for curve in stellar_var])
            return stellar_var

        # Add physics-based transits
        hybrid_curves = []
        for i in range(num_samples):
            # Start with VAE variability
            curve = stellar_var[i].copy()

            # Convert to flux units (mean=1)
            curve = curve * 0.01 + 1.0

            # Add transits
            time = np.linspace(0, 1, len(curve))
            n_transits = num_transits if num_transits else np.random.randint(2, 5)

            for _ in range(n_transits):
                period = np.random.uniform(0.15, 0.4)
                t0 = np.random.uniform(0, period)
                depth = np.random.uniform(0.005, 0.03)
                duration = np.random.uniform(0.01, 0.03)

                transit = self.physics_gen.generate_transit(time, period, t0, depth, duration)
                curve *= transit

            # Add basic photometric noise
            curve = self.physics_gen.add_noise(curve, 0.001)

            # Renormalize
            curve = (curve - np.mean(curve)) / np.std(curve)
            
            # Add realistic Kepler-like noise AFTER normalization
            if add_realistic_noise:
                curve = self.add_realistic_noise(curve, noise_level)
                # Re-normalize to ensure mean=0, std=1
                curve = (curve - np.mean(curve)) / np.std(curve)
            
            hybrid_curves.append(curve)

        return np.array(hybrid_curves)

print("Hybrid generator defined")

# CELL 7: Training Function
def train_hybrid_system(data_folder, num_epochs=50, batch_size=32):
    """
    Train the hybrid system

    Returns:
        HybridLightCurveGenerator ready to generate new curves
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print("=" * 70)

    # Load data
    print("\n1. Loading data and extracting stellar variability...")
    dataset = HybridDataset(data_folder, target_length=2000)

    if len(dataset) == 0:
        raise ValueError("No data loaded! Check your data folder.")

    # Remove transits to get pure stellar variability
    print("\n2. Removing transits to isolate stellar variability...")
    stellar_var_data = []
    for i in range(len(dataset)):
        curve = dataset[i].numpy()
        clean_curve = remove_transits(curve)
        stellar_var_data.append(clean_curve)

    print(f"✓ Extracted stellar variability from {len(stellar_var_data)} curves")

    # Create dataset
    stellar_var_tensor = torch.stack([torch.FloatTensor(c) for c in stellar_var_data])
    dataloader = DataLoader(stellar_var_tensor, batch_size=batch_size, shuffle=True)

    # Initialize VAE
    print("\n3. Training VAE on stellar variability...")
    vae = StellarVAE(input_dim=2000, latent_dim=64).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    total_params = sum(p.numel() for p in vae.parameters())
    print(f"VAE parameters: {total_params:,}")
    print("=" * 70)

    import time
    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        vae.train()
        epoch_loss = 0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = vae(batch)

            # VAE loss
            recon_loss = F.mse_loss(recon, batch, reduction='sum') / batch.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
            loss = recon_loss + 0.1 * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (num_epochs - epoch - 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} - Loss: {epoch_loss/len(dataloader):7.4f} - "
                  f"Time: {elapsed/60:4.1f}m, ETA: {eta/60:4.1f}m")

    print(f"\n✓ Training complete! Total time: {elapsed/60:.1f} minutes")

    # Create hybrid generator
    print("\n4. Creating hybrid generator...")
    hybrid_gen = HybridLightCurveGenerator(vae, device)

    # Generate test samples
    print("\n5. Generating test samples...")
    test_samples = hybrid_gen.generate(num_samples=8, add_transits=True)

    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i in range(8):
        time = np.linspace(0, 1, len(test_samples[i]))
        axes[i].plot(time, test_samples[i], 'b-', linewidth=1, alpha=0.8)
        axes[i].set_xlabel('Normalized Time', fontsize=10)
        axes[i].set_ylabel('Normalized Flux', fontsize=10)
        axes[i].set_title(f'Hybrid Light Curve {i+1}', fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_facecolor('#f8f9fa')

        # Add statistics
        stats = f'μ: {np.mean(test_samples[i]):.3f}\nσ: {np.std(test_samples[i]):.3f}'
        axes[i].text(0.02, 0.98, stats, transform=axes[i].transAxes,
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('hybrid_lightcurves.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Saved samples to: hybrid_lightcurves.png")
    print("=" * 70)
    print("\nHybrid generator ready!")
    print("Usage: new_curves = hybrid_gen.generate(num_samples=100)")

    return hybrid_gen

print("Training function defined")

# CELL 8: Train and Generate
# Train the system
print("\n" + "=" * 70)
print("TRAINING HYBRID SYSTEM")
print("=" * 70)

DATA_FOLDER = "/content/light_curves"

hybrid_gen = train_hybrid_system(DATA_FOLDER, num_epochs=50, batch_size=32)

print("\n" + "=" * 70)
print("GENERATING SYNTHETIC LIGHT CURVES WITH REALISTIC NOISE")
print("=" * 70)

# Generate 100 new curves with realistic noise
num_to_generate = 100
print(f"\nGenerating {num_to_generate} synthetic light curves with Kepler-like noise...")

# Generate with realistic noise (noise_level=0.3 matches real Kepler statistics)
synthetic_curves = hybrid_gen.generate(
    num_samples=num_to_generate, 
    add_transits=True,
    noise_level=0.3,  # Adjust this to match your real data (0.2-0.4 typical)
    add_realistic_noise=True
)

print(f"  Generated {num_to_generate} curves")
print(f"  Shape: {synthetic_curves.shape}")
print(f"  Mean: {np.mean(synthetic_curves):.4f}")
print(f"  Std: {np.std(synthetic_curves):.4f}")

# Calculate point-to-point scatter to verify noise
ptp_scatter = np.mean([np.std(np.diff(curve)) for curve in synthetic_curves])
print(f"  Point-to-point scatter: {ptp_scatter:.4f} (target: ~0.9)")

# Save to file
np.save('synthetic_kepler_lightcurves.npy', synthetic_curves)
print(f"Saved to: synthetic_kepler_lightcurves.npy")

# Show a few examples
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i in range(6):
    idx = np.random.randint(0, num_to_generate)
    time = np.linspace(0, 1, len(synthetic_curves[idx]))
    axes[i].plot(time, synthetic_curves[idx], 'b-', linewidth=0.8, alpha=0.9)
    axes[i].set_xlabel('Normalized Time')
    axes[i].set_ylabel('Normalized Flux')
    axes[i].set_title(f'Synthetic Curve {idx+1} (with noise)')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('synthetic_examples_with_noise.png', dpi=200, bbox_inches='tight')
plt.show()

print("Saved examples to: synthetic_examples_with_noise.png")

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print("\nNoise level can be adjusted via:")
print("  synthetic_curves = hybrid_gen.generate(num_samples=100, noise_level=0.3)")
print("  - noise_level=0.2: Less noisy")
print("  - noise_level=0.3: Matches typical Kepler (recommended)")
print("  - noise_level=0.4: More noisy")
print("=" * 70)


# STATISTICAL ANALYSIS

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
import os
import glob
import matplotlib.pyplot as plt

# ==================== DATA LOADING ====================

def load_synthetic_data(filepath='synthetic_kepler_lightcurves.npy'):
    """Load synthetic light curves from .npy file"""
    data = np.load(filepath)
    print(f"Loaded {len(data)} synthetic light curves")
    return data

def load_real_data(folder_path='/content/light_curves'):
    """Load all real Kepler light curves from CSV files"""
    # Try multiple possible paths
    possible_paths = [
        folder_path,
        'content/light_curves',
        './content/light_curves',
        'light_curves'
    ]
    
    csv_files = []
    for path in possible_paths:
        csv_files = glob.glob(os.path.join(path, '*.csv'))
        if csv_files:
            print(f"Found {len(csv_files)} CSV files in: {path}")
            break
    
    if not csv_files:
        print(f"ERROR: No CSV files found in any of these locations:")
        for path in possible_paths:
            print(f"   - {path}")
        print("\nPlease check your folder path!")
        return []
    
    light_curves = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        # Try to find flux column (adjust column name if needed)
        if 'flux' in df.columns:
            flux = df['flux'].values
        elif 'PDCSAP_FLUX' in df.columns:
            flux = df['PDCSAP_FLUX'].values
        elif 'SAP_FLUX' in df.columns:
            flux = df['SAP_FLUX'].values
        else:
            # Assume second column if flux column not found
            flux = df.iloc[:, 1].values
        
        # Remove NaNs
        flux = flux[~np.isnan(flux)]
        if len(flux) > 0:
            # Clip outliers using sigma clipping (commonly used in astronomy)
            median = np.median(flux)
            mad = np.median(np.abs(flux - median))
            sigma = 1.4826 * mad  # Convert MAD to standard deviation
            
            # Clip at 5-sigma (adjust threshold if needed)
            lower_bound = median - 5 * sigma
            upper_bound = median + 5 * sigma
            flux_clipped = np.clip(flux, lower_bound, upper_bound)
            
            # Normalize to match synthetic data (mean=0, std=1)
            flux_normalized = (flux_clipped - np.mean(flux_clipped)) / np.std(flux_clipped)
            light_curves.append(flux_normalized)
    
    print(f"✓ Loaded {len(light_curves)} real light curves")
    return light_curves

# ==================== FEATURE EXTRACTION ====================

def compute_features(light_curve):
    """Extract statistical features from a single light curve"""
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(light_curve)
    features['std'] = np.std(light_curve)
    features['median'] = np.median(light_curve)
    features['skewness'] = stats.skew(light_curve)
    features['kurtosis'] = stats.kurtosis(light_curve)
    
    # Variability measures
    features['range'] = np.ptp(light_curve)
    features['iqr'] = np.percentile(light_curve, 75) - np.percentile(light_curve, 25)
    
    # Point-to-point scatter (important for light curve quality)
    if len(light_curve) > 1:
        features['point_to_point_scatter'] = np.std(np.diff(light_curve))
    
    # Autocorrelation at lag 1 (measures temporal correlation)
    if len(light_curve) > 1:
        features['autocorr_lag1'] = np.corrcoef(light_curve[:-1], light_curve[1:])[0, 1]
    
    return features

def compute_all_features(light_curves):
    """Compute features for all light curves"""
    all_features = {key: [] for key in compute_features(light_curves[0]).keys()}
    
    for lc in light_curves:
        features = compute_features(lc)
        for key, value in features.items():
            all_features[key].append(value)
    
    return {key: np.array(values) for key, values in all_features.items()}

# ==================== STATISTICAL TESTS ====================

def wasserstein_distance(data1, data2):
    """Compute Wasserstein distance (Earth Mover's Distance)"""
    return stats.wasserstein_distance(data1, data2)

def ks_test(data1, data2):
    """Perform Kolmogorov-Smirnov test"""
    statistic, pvalue = stats.ks_2samp(data1, data2)
    return statistic, pvalue

def jensen_shannon_divergence(data1, data2, bins=50):
    """Compute Jensen-Shannon divergence"""
    range_min = min(data1.min(), data2.min())
    range_max = max(data1.max(), data2.max())
    
    hist1, _ = np.histogram(data1, bins=bins, range=(range_min, range_max), density=True)
    hist2, _ = np.histogram(data2, bins=bins, range=(range_min, range_max), density=True)
    
    # Normalize
    hist1 = hist1 / (hist1.sum() + 1e-10)
    hist2 = hist2 / (hist2.sum() + 1e-10)
    
    # Add small epsilon to avoid log(0)
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    
    return jensenshannon(hist1, hist2)

def compare_distributions(synthetic_features, real_features):
    """Compare feature distributions with multiple metrics"""
    results = []
    
    for feature_name in synthetic_features.keys():
        synth_data = synthetic_features[feature_name]
        real_data = real_features[feature_name]
        
        # Remove NaNs
        synth_data = synth_data[~np.isnan(synth_data)]
        real_data = real_data[~np.isnan(real_data)]
        
        # Compute statistics
        synth_mean = np.mean(synth_data)
        synth_std = np.std(synth_data)
        real_mean = np.mean(real_data)
        real_std = np.std(real_data)
        
        # Statistical tests
        ks_stat, ks_pval = ks_test(synth_data, real_data)
        wasserstein = wasserstein_distance(synth_data, real_data)
        js_div = jensen_shannon_divergence(synth_data, real_data)
        
        results.append({
            'Feature': feature_name,
            'Synthetic_Mean': synth_mean,
            'Synthetic_Std': synth_std,
            'Real_Mean': real_mean,
            'Real_Std': real_std,
            'KS_statistic': ks_stat,
            'KS_pvalue': ks_pval,
            'Wasserstein': wasserstein,
            'JS_Divergence': js_div
        })
    
    return pd.DataFrame(results)

# ==================== REPORTING ====================

def print_report(df_results):
    """Print a formatted statistical report"""
    print("\n" + "="*100)
    print("STATISTICAL VALIDATION: SYNTHETIC vs REAL KEPLER LIGHT CURVES")
    print("="*100 + "\n")
    
    for _, row in df_results.iterrows():
        print(f"{row['Feature'].upper().replace('_', ' ')}")
        print(f"   Synthetic: {row['Synthetic_Mean']:.6f} ± {row['Synthetic_Std']:.6f}")
        print(f"   Real:      {row['Real_Mean']:.6f} ± {row['Real_Std']:.6f}")
        print(f"   KS test p-value:      {row['KS_pvalue']:.4f} {'✓ PASS' if row['KS_pvalue'] > 0.05 else '✗ FAIL'}")
        print(f"   Wasserstein distance: {row['Wasserstein']:.6f}")
        print(f"   JS divergence:        {row['JS_Divergence']:.6f}")
        print()
    
    # Overall assessment
    print("="*100)
    print("OVERALL ASSESSMENT")
    print("="*100)
    avg_pval = df_results['KS_pvalue'].mean()
    avg_wasserstein = df_results['Wasserstein'].mean()
    avg_js = df_results['JS_Divergence'].mean()
    
    print(f"Average KS p-value:        {avg_pval:.4f}")
    print(f"Average Wasserstein dist:  {avg_wasserstein:.6f}")
    print(f"Average JS divergence:     {avg_js:.6f}")
    
    # Interpretation
    passing = (df_results['KS_pvalue'] > 0.05).sum()
    total = len(df_results)
    print(f"\nFeatures passing KS test (p > 0.05): {passing}/{total} ({100*passing/total:.1f}%)")
    
    print("\n" + "="*100)
    if avg_pval > 0.05 and avg_js < 0.1:
        print("EXCELLENT: Synthetic light curves are statistically indistinguishable from real data!")
    elif avg_pval > 0.05:
        print("GOOD: Most features match well, minor differences exist.")
    elif avg_pval > 0.01:
        print("MODERATE: Some statistical differences detected. Consider refinement.")
    else:
        print("POOR: Significant differences detected. Model needs improvement.")
    print("="*100)

# ==================== VISUALIZATION ====================

def plot_comparison(synthetic_features, real_features):
    """Create visualization comparing distributions"""
    n_features = len(synthetic_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature_name in enumerate(synthetic_features.keys()):
        ax = axes[idx]
        synth_data = synthetic_features[feature_name]
        real_data = real_features[feature_name]
        
        # Remove NaNs
        synth_data = synth_data[~np.isnan(synth_data)]
        real_data = real_data[~np.isnan(real_data)]
        
        ax.hist(real_data, bins=30, alpha=0.6, label='Real', density=True, color='#2E86AB')
        ax.hist(synth_data, bins=30, alpha=0.6, label='Synthetic', density=True, color='#A23B72')
        ax.set_xlabel(feature_name.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# ==================== MAIN EXECUTION ====================

print("Starting Statistical Validation Analysis")

# Load data
synthetic_lcs = load_synthetic_data()
real_lcs = load_real_data()

# Check if we have real data
if len(real_lcs) == 0:
    print("ERROR: No real light curves loaded.")
    print("Please check that your CSV files are in the correct folder.")
else:
    # Compute features
    print("Computing statistical features")
    synthetic_features = compute_all_features(synthetic_lcs)
    real_features = compute_all_features(real_lcs)

    # Run comparisons
    print("Running statistical tests")
    results_df = compare_distributions(synthetic_features, real_features)

    # Display results
    print_report(results_df)

    # Visualize
    print("\nGenerating comparison plots...")
    plot_comparison(synthetic_features, real_features)

    # Save results
    results_df.to_csv('statistical_validation_results.csv', index=False)
    print("\nDetailed results saved to: statistical_validation_results.csv")
    print("Analysis complete!")

# PARAMETER INFERENCE NEURAL NETWORK

pip install torch torchvision numpy matplotlib batman-package

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import batman

class TransitDataGenerator:
    """Generate synthetic transit light curves using batman (Mandel-Agol)"""

    def __init__(self, n_points: int = 200):
        self.n_points = n_points

    def generate_light_curve(self, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a transit light curve from parameters."""
        # Set up batman parameters
        batman_params = batman.TransitParams()
        batman_params.t0 = params['t0']
        batman_params.per = params['per']
        batman_params.rp = params['rp_rs']
        batman_params.a = params['a_rs']
        batman_params.inc = params['inc']
        batman_params.ecc = 0.0
        batman_params.w = 90.0
        batman_params.u = [params['u1'], params['u2']]
        batman_params.limb_dark = "quadratic"

        # Generate time array centered on transit
        duration = params['per'] / 10
        t = np.linspace(params['t0'] - duration/2, params['t0'] + duration/2, self.n_points)

        # Generate the transit model
        m = batman.TransitModel(batman_params, t)
        flux = m.light_curve(batman_params)

        # Add noise if specified
        if 'noise' in params and params['noise'] > 0:
            flux += np.random.normal(0, params['noise'], size=len(flux))

        return t, flux

    def sample_parameters(self, noise_level: float = 0.001) -> Dict[str, float]:
        """Sample random but physically reasonable transit parameters"""

        rp_rs = np.random.uniform(0.01, 0.2)
        # More focused range for semi-major axis (easier to learn)
        a_rs = np.random.uniform(5, 30)
        inc = np.random.uniform(85, 90)
        # More focused period range
        per = np.random.uniform(1.0, 15.0)
        t0 = 0.0
        u1 = np.random.uniform(0.1, 0.6)
        u2 = np.random.uniform(0.1, 0.5)

        return {
            'rp_rs': rp_rs, 'a_rs': a_rs, 'inc': inc, 't0': t0,
            'per': per, 'u1': u1, 'u2': u2, 'noise': noise_level
        }


class TransitDataset(Dataset):
    """PyTorch Dataset for transit light curves"""

    def __init__(self, n_samples: int, n_points: int = 200, noise_level: float = 0.001):
        self.n_samples = n_samples
        self.generator = TransitDataGenerator(n_points)
        self.noise_level = noise_level
        self.n_points = n_points

        # Pre-generate all data
        print(f"Generating {n_samples} transit light curves...")
        self.data = []
        for i in range(n_samples):
            params = self.generator.sample_parameters(noise_level)
            t, flux = self.generator.generate_light_curve(params)

            # Store raw parameters for normalization
            self.data.append({
                'time': t,
                'flux': flux,
                'limb_darkening': np.array([params['u1'], params['u2']]),
                'targets': np.array([
                    params['rp_rs'],
                    params['a_rs'],
                    params['inc'],
                    params['per']
                ])
            })

            if (i + 1) % 2000 == 0:
                print(f"  Generated {i + 1}/{n_samples}")

        # Compute normalization statistics for targets
        all_targets = np.array([d['targets'] for d in self.data])
        self.target_mean = all_targets.mean(axis=0)
        self.target_std = all_targets.std(axis=0)
        print(f"  Target normalization - Mean: {self.target_mean}, Std: {self.target_std}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Normalize flux for network input
        flux_norm = (sample['flux'] - np.mean(sample['flux'])) / np.std(sample['flux'])

        # Normalize targets to similar scale (helps training)
        targets_norm = (sample['targets'] - self.target_mean) / self.target_std

        return (
            torch.FloatTensor(flux_norm),
            torch.FloatTensor(sample['flux']),
            torch.FloatTensor(sample['limb_darkening']),
            torch.FloatTensor(targets_norm),  # normalized
            torch.FloatTensor(sample['targets'])  # original
        )


class TransitNet(nn.Module):
    """Neural network to predict transit parameters from light curves."""

    def __init__(self, n_points: int = 200):
        super(TransitNet, self).__init__()

        # Deeper 1D CNN for better feature extraction
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # Calculate size after convolutions and pooling
        conv_output_size = n_points // 16  # 4 pooling layers
        flattened_size = 256 * conv_output_size

        # Larger dense layers
        self.fc1 = nn.Linear(flattened_size + 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)  # Output: rp_rs, a_rs, inc, per

        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(256)

    def forward(self, flux, limb_darkening):
        x = flux.unsqueeze(1)

        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(self.relu(self.batch_norm4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = torch.cat([x, limb_darkening], dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def train_model(model, train_loader, val_loader, n_epochs=100, lr=0.001, device='cpu'):
    """Train model using parameter loss with normalization"""

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0

    print("\nTraining with normalized parameter MSE loss...")
    print("Using larger network and more epochs for better performance")

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0

        for flux_norm, flux_orig, limb_darkening, targets_norm, targets_orig in train_loader:
            flux_norm = flux_norm.to(device)
            limb_darkening = limb_darkening.to(device)
            targets_norm = targets_norm.to(device)

            optimizer.zero_grad()
            outputs = model(flux_norm, limb_darkening)
            loss = criterion(outputs, targets_norm)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for flux_norm, flux_orig, limb_darkening, targets_norm, targets_orig in val_loader:
                flux_norm = flux_norm.to(device)
                limb_darkening = limb_darkening.to(device)
                targets_norm = targets_norm.to(device)

                outputs = model(flux_norm, limb_darkening)
                loss = criterion(outputs, targets_norm)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping after 20 epochs of no improvement
        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return history


# Example usage
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate datasets (larger for better learning)
    n_points = 200
    print("\nGenerating training data...")
    train_dataset = TransitDataset(n_samples=8000, n_points=n_points, noise_level=0.001)
    val_dataset = TransitDataset(n_samples=1000, n_points=n_points, noise_level=0.001)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Create and train model
    model = TransitNet(n_points=n_points)
    print(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")

    history = train_model(
        model, train_loader, val_loader,
        n_epochs=100, lr=0.001, device=device
    )

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training History (Parameter MSE)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\nTraining complete! Saved training history to training_history.png")

    # Save model
    torch.save(model.state_dict(), 'transit_model.pth')
    print("Model saved to transit_model.pth")

    # Test on a few examples with batman reconstruction
    print("\nTesting on validation examples...")
    model.eval()

    n_test = 10
    errors = {'rp_rs': [], 'a_rs': [], 'inc': [], 'per': []}

    with torch.no_grad():
        for i in range(n_test):
            flux_norm, flux_orig, limb_darkening, targets_norm, targets_orig = val_dataset[i]
            flux_norm = flux_norm.unsqueeze(0).to(device)
            limb_darkening_batch = limb_darkening.unsqueeze(0).to(device)

            # Predict and denormalize
            prediction_norm = model(flux_norm, limb_darkening_batch)
            prediction = prediction_norm.cpu().numpy()[0] * val_dataset.target_std + val_dataset.target_mean
            targets = targets_orig.numpy()

            errors['rp_rs'].append(abs(targets[0] - prediction[0]) / targets[0] * 100)  # % error
            errors['a_rs'].append(abs(targets[1] - prediction[1]) / targets[1] * 100)
            errors['inc'].append(abs(targets[2] - prediction[2]))
            errors['per'].append(abs(targets[3] - prediction[3]) / targets[3] * 100)

    print(f"\nAverage errors over {n_test} test examples:")
    print(f"  Rp/Rs: {np.mean(errors['rp_rs']):.2f}% ± {np.std(errors['rp_rs']):.2f}%")
    print(f"  a/Rs:  {np.mean(errors['a_rs']):.2f}% ± {np.std(errors['a_rs']):.2f}%")
    print(f"  inc:   {np.mean(errors['inc']):.3f}° ± {np.std(errors['inc']):.3f}°")
    print(f"  per:   {np.mean(errors['per']):.2f}% ± {np.std(errors['per']):.2f}%")

    # Visualize one example with batman reconstruction
    print("\nGenerating reconstruction visualization...")
    flux_norm, flux_orig, limb_darkening, targets_norm, targets_orig = val_dataset[0]
    flux_norm = flux_norm.unsqueeze(0).to(device)
    limb_darkening_batch = limb_darkening.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_norm = model(flux_norm, limb_darkening_batch)
        prediction = prediction_norm.cpu().numpy()[0] * val_dataset.target_std + val_dataset.target_mean
    targets = targets_orig.numpy()

    # Generate reconstructed light curve using batman
    batman_params = batman.TransitParams()
    batman_params.t0 = 0.0
    batman_params.per = prediction[3]
    batman_params.rp = prediction[0]
    batman_params.a = prediction[1]
    batman_params.inc = prediction[2]
    batman_params.ecc = 0.0
    batman_params.w = 90.0
    batman_params.u = limb_darkening.numpy().tolist()
    batman_params.limb_dark = "quadratic"

    duration = prediction[3] / 10
    t = np.linspace(-duration/2, duration/2, n_points)
    m = batman.TransitModel(batman_params, t)
    reconstructed_flux = m.light_curve(batman_params)

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t, flux_orig.numpy(), 'k.', label='True Light Curve', alpha=0.6, markersize=4)
    plt.plot(t, reconstructed_flux, 'r-', label='Reconstructed from Prediction', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.title('Light Curve Reconstruction')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    params = ['Rp/Rs', 'a/Rs', 'inc (°)', 'per (d)']
    x = np.arange(len(params))
    width = 0.35
    plt.bar(x - width/2, targets, width, label='True', alpha=0.7)
    plt.bar(x + width/2, prediction, width, label='Predicted', alpha=0.7)
    plt.xlabel('Parameter')
    plt.ylabel('Value')
    plt.title('Parameter Comparison')
    plt.xticks(x, params)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('reconstruction_example.png', dpi=150)
    print("Saved reconstruction example to reconstruction_example.png")

    print("\n" + "="*50)
    print("Example prediction details:")
    print(f"  Rp/Rs - True: {targets[0]:.4f}, Pred: {prediction[0]:.4f}, Error: {abs(targets[0]-prediction[0]):.4f}")
    print(f"  a/Rs  - True: {targets[1]:.4f}, Pred: {prediction[1]:.4f}, Error: {abs(targets[1]-prediction[1]):.4f}")
    print(f"  inc   - True: {targets[2]:.4f}°, Pred: {prediction[2]:.4f}°, Error: {abs(targets[2]-prediction[2]):.4f}°")
    print(f"  per   - True: {targets[3]:.4f} d, Pred: {prediction[3]:.4f} d, Error: {abs(targets[3]-prediction[3]):.4f} d")


# COMPARISON WITH MCMC

!pip install emcee corner

import emcee
import corner
from scipy.optimize import minimize
import time as time_module

n_points = 192

# Load  trained model
device = torch.device('cpu')
model = TransitNet(n_points=n_points)
model.load_state_dict(torch.load('transit_model.pth', map_location=device))
model.eval()
print("Model loaded successfully")

# Generate validation dataset with matching n_points
np.random.seed(123)
val_dataset = TransitDataset(n_samples=100, n_points=n_points, noise_level=0.001)
print("Dataset ready")

# MCMC likelihood function
def log_likelihood(params, time, flux, flux_err, u1, u2):
    """Log likelihood for MCMC"""
    rp_rs, a_rs, inc, per = params
    
    # Priors
    if not (0.01 < rp_rs < 0.3): return -np.inf
    if not (3 < a_rs < 40): return -np.inf
    if not (80 < inc < 90): return -np.inf
    if not (0.5 < per < 20): return -np.inf
    
    # Generate model
    batman_params = batman.TransitParams()
    batman_params.t0 = 0.0
    batman_params.per = per
    batman_params.rp = rp_rs
    batman_params.a = a_rs
    batman_params.inc = inc
    batman_params.ecc = 0.0
    batman_params.w = 90.0
    batman_params.u = [u1, u2]
    batman_params.limb_dark = "quadratic"
    
    try:
        m = batman.TransitModel(batman_params, time)
        model_flux = m.light_curve(batman_params)
    except:
        return -np.inf
    
    chi2 = np.sum(((flux - model_flux) / flux_err) ** 2)
    return -0.5 * chi2


def compare_methods(test_idx, model, val_dataset, n_mcmc_steps=2000):
    """Compare MCMC and NN on a single light curve"""
    
    # Get test data
    flux_norm, flux_orig, limb_darkening, targets_norm, targets_orig = val_dataset[test_idx]
    
    time_array = val_dataset.data[test_idx]['time']
    flux = flux_orig.numpy()
    limb_dark = limb_darkening.numpy()
    true_params = targets_orig.numpy()
    
    flux_err = np.std(flux - np.median(flux)) * np.ones_like(flux)
    
    print(f"\nTest example {test_idx}")
    print(f"True: Rp/Rs={true_params[0]:.4f}, a/Rs={true_params[1]:.2f}, inc={true_params[2]:.2f}°, per={true_params[3]:.2f}d")
    print("="*70)
    
    # Run MCMC
    print("Running MCMC...")
    start = time_module.time()
    
    # Optimize first
    def neg_log_like(params):
        return -log_likelihood(params, time_array, flux, flux_err, limb_dark[0], limb_dark[1])
    
    initial = true_params + np.random.randn(4) * 0.01
    result = minimize(neg_log_like, initial, method='Nelder-Mead')
    best_fit = result.x
    
    # MCMC
    n_walkers, ndim = 32, 4
    pos = best_fit + 1e-3 * np.random.randn(n_walkers, ndim)
    
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_likelihood,
        args=(time_array, flux, flux_err, limb_dark[0], limb_dark[1])
    )
    
    sampler.run_mcmc(pos, 500, progress=False)
    sampler.reset()
    sampler.run_mcmc(None, n_mcmc_steps, progress=False)
    
    mcmc_time = time_module.time() - start
    
    samples = sampler.get_chain(discard=500, thin=10, flat=True)
    mcmc_params = np.median(samples, axis=0)
    mcmc_stds = np.std(samples, axis=0)
    
    print(f"  MCMC done in {mcmc_time:.2f}s")
    
    # Run NN
    print("Running NN...")
    start = time_module.time()
    
    model.eval()
    with torch.no_grad():
        flux_norm_tensor = torch.FloatTensor(flux_norm).unsqueeze(0).to(device)
        limb_tensor = torch.FloatTensor(limb_dark).unsqueeze(0).to(device)
        
        pred_norm = model(flux_norm_tensor, limb_tensor)
        nn_params = pred_norm.cpu().numpy()[0] * val_dataset.target_std + val_dataset.target_mean
    
    nn_time = time_module.time() - start
    print(f"  NN done in {nn_time:.4f}s")
    
    # Compute errors
    mcmc_errors = np.abs(mcmc_params - true_params) / true_params * 100
    nn_errors = np.abs(nn_params - true_params) / true_params * 100
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    param_names = ['Rp/Rs', 'a/Rs', 'inc', 'period']
    print(f"\n{'Param':<8} {'True':<10} {'MCMC':<15} {'MCMC Err':<10} {'NN':<10} {'NN Err':<10}")
    print("-"*70)
    
    for i, name in enumerate(param_names):
        print(f"{name:<8} {true_params[i]:>8.4f} {mcmc_params[i]:>8.4f}±{mcmc_stds[i]:.3f} "
              f"{mcmc_errors[i]:>8.2f}% {nn_params[i]:>8.4f} {nn_errors[i]:>8.2f}%")
    
    print(f"\nTime: MCMC={mcmc_time:.2f}s, NN={nn_time:.4f}s, Speedup={mcmc_time/nn_time:.0f}x")
    
    return {
        'true_params': true_params, 'mcmc_params': mcmc_params, 'mcmc_stds': mcmc_stds,
        'mcmc_errors': mcmc_errors, 'mcmc_time': mcmc_time, 'nn_params': nn_params,
        'nn_errors': nn_errors, 'nn_time': nn_time, 'flux': flux, 'time': time_array,
        'limb_dark': limb_dark, 'mcmc_samples': samples
    }


# RUN COMPARISON
print("\n" + "="*70)
print("MCMC vs Neural Network Comparison")
print("="*70)
# Run on 10 examples
all_results = []
for i in range(10):
    results = compare_methods(i, model, val_dataset, n_mcmc_steps=2000)
    all_results.append(results)

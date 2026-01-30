#!/usr/bin/env python3
"""
Model fitting and comparison script for MPI ring benchmark.
Fits Alpha-Beta, LogP, and LogGP model parameters from measured data,
then plots measured vs predicted round-trip times.

Usage: python3 plot_model_vs_measured.py [results_file] [num_procs]
  Default: results_4.dat with P=4

Expects data file columns: bytes time_us
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# --- Configuration ---
data_file = sys.argv[1] if len(sys.argv) > 1 else "results_4.dat"
P = int(sys.argv[2]) if len(sys.argv) > 2 else 4

if not os.path.exists(data_file):
    print(f"Error: {data_file} not found. Run step4 first.")
    sys.exit(1)

data = np.loadtxt(data_file, comments='#')
if data.ndim == 1:
    data = data.reshape(1, -1)

bytes_arr = data[:, 0]
time_us = data[:, 1]  # round-trip time in microseconds

# We need at least 2 data points for fitting
if len(bytes_arr) < 2:
    print("Need at least 2 data points for model fitting.")
    sys.exit(1)

# =============================================================================
# Model definitions for a ring of P processes
#
# In a ring, the message traverses P hops sequentially.
#
# Alpha-Beta model:
#   T(s, P) = P * (alpha + beta * s)
#   where alpha = latency per hop, beta = inverse bandwidth (us/byte)
#
# LogP model (small/fixed-size messages):
#   T(P) = P * (L + 2*o)
#   where L = network latency, o = CPU overhead
#   (each hop: sender overhead o, network latency L, receiver overhead o)
#   Since g (gap) only matters for back-to-back sends from same rank,
#   and each rank sends exactly once in a ring, g doesn't appear.
#
# LogGP model (variable-size messages):
#   T(s, P) = P * (L + 2*o + (s-1)*G)   for s >= 1
#   T(0, P) = P * (L + 2*o)              for s = 0
#   where G = gap per byte for long messages
# =============================================================================

# --- Fit Alpha-Beta: T = P * (alpha + beta * s) ---
# Rearrange: T/P = alpha + beta * s
# Linear fit on (s, T/P)
per_hop = time_us / P
coeffs_ab = np.polyfit(bytes_arr, per_hop, 1)
beta_ab = coeffs_ab[0]   # us/byte per hop
alpha_ab = coeffs_ab[1]  # us per hop

print("=== Alpha-Beta Model ===")
print(f"  alpha (latency/hop)     = {alpha_ab:.4f} us")
print(f"  beta  (inv bandwidth)   = {beta_ab:.6f} us/byte")
print(f"  => T(s, P) = P * ({alpha_ab:.4f} + {beta_ab:.6f} * s)")

# --- Fit LogP: use smallest message size ---
# T = P * (L + 2*o)
# We can't separately identify L and o from one measurement,
# so define: logp_hop = L + 2*o = T_small / P
small_idx = 0  # index of smallest message
logp_hop = time_us[small_idx] / P

print(f"\n=== LogP Model ===")
print(f"  L + 2*o (per hop)       = {logp_hop:.4f} us")
print(f"  (from {int(bytes_arr[small_idx])}-byte message)")
print(f"  => T(P) = P * {logp_hop:.4f}  (fixed packet size)")

# --- Fit LogGP: T = P * (L + 2*o + max(0, s-1)*G) ---
# Use LogP's (L+2o) from small messages, fit G from large messages
# T/P = (L+2o) + (s-1)*G  for s >= 1
# (T/P - logp_hop) = (s-1)*G
mask_large = bytes_arr > 0
if np.sum(mask_large) >= 2:
    s_large = bytes_arr[mask_large]
    t_large = per_hop[mask_large]
    # Fit: t_large = logp_hop + (s-1)*G
    # => (t_large - logp_hop) = G * (s - 1)
    s_minus_1 = s_large - 1
    # Only use points where s > 1 for a clean fit
    mask_fit = s_minus_1 > 0
    if np.sum(mask_fit) >= 1:
        G_fit = np.polyfit(s_minus_1[mask_fit],
                           (t_large[mask_fit] - logp_hop), 1)
        G_loggp = G_fit[0]
    else:
        G_loggp = beta_ab  # fallback
else:
    G_loggp = beta_ab  # fallback

print(f"\n=== LogGP Model ===")
print(f"  L + 2*o (per hop)       = {logp_hop:.4f} us")
print(f"  G       (gap/byte)      = {G_loggp:.6f} us/byte")
print(f"  => T(s, P) = P * ({logp_hop:.4f} + max(0, s-1) * {G_loggp:.6f})")

# --- Generate predictions ---
s_plot = np.linspace(0, max(bytes_arr) * 1.05, 500)

t_ab = P * (alpha_ab + beta_ab * s_plot)
t_logp = np.full_like(s_plot, P * logp_hop)
t_loggp = P * (logp_hop + np.maximum(0, s_plot - 1) * G_loggp)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: linear scale
ax = axes[0]
ax.plot(bytes_arr, time_us, 'ko', markersize=8, label='Measured', zorder=5)
ax.plot(s_plot, t_ab, '-', linewidth=2, label=f'Alpha-Beta', color='tab:blue')
ax.plot(s_plot, t_logp, '--', linewidth=2, label=f'LogP', color='tab:orange')
ax.plot(s_plot, t_loggp, '-.', linewidth=2, label=f'LogGP', color='tab:green')
ax.set_xlabel('Message size (bytes)', fontsize=12)
ax.set_ylabel('Round-trip time (us)', fontsize=12)
ax.set_title(f'Ring Models vs Measured (P={P}, linear scale)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Right: log-log scale (skip 0-byte for log)
ax = axes[1]
mask_pos = bytes_arr > 0
s_plot_log = s_plot[s_plot > 0]
t_ab_log = P * (alpha_ab + beta_ab * s_plot_log)
t_logp_log = np.full_like(s_plot_log, P * logp_hop)
t_loggp_log = P * (logp_hop + np.maximum(0, s_plot_log - 1) * G_loggp)

ax.loglog(bytes_arr[mask_pos], time_us[mask_pos], 'ko', markersize=8,
          label='Measured', zorder=5)
ax.loglog(s_plot_log, t_ab_log, '-', linewidth=2, label='Alpha-Beta',
          color='tab:blue')
ax.loglog(s_plot_log, t_logp_log, '--', linewidth=2, label='LogP',
          color='tab:orange')
ax.loglog(s_plot_log, t_loggp_log, '-.', linewidth=2, label='LogGP',
          color='tab:green')
ax.set_xlabel('Message size (bytes)', fontsize=12)
ax.set_ylabel('Round-trip time (us)', fontsize=12)
ax.set_title(f'Ring Models vs Measured (P={P}, log-log scale)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
print(f"\nPlot saved to model_comparison.png")

# --- Print model error summary ---
print("\n=== Model Errors (RMS % error) ===")
for name, pred_fn in [
    ("Alpha-Beta", lambda s: P * (alpha_ab + beta_ab * s)),
    ("LogP",       lambda s: P * logp_hop * np.ones_like(s)),
    ("LogGP",      lambda s: P * (logp_hop + np.maximum(0, s - 1) * G_loggp)),
]:
    pred = pred_fn(bytes_arr)
    # Avoid division by zero
    nonzero = time_us > 0
    pct_err = np.sqrt(np.mean(((pred[nonzero] - time_us[nonzero]) / time_us[nonzero]) ** 2)) * 100
    print(f"  {name:12s}: {pct_err:.1f}%")

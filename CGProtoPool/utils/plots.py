"""
Plotting functions for the prototype pool
"""

import numpy as np
import matplotlib.pyplot as plt

# CUB plots -------------------------------------------------------------------------------------
cub_n = [25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 1000, 1250, 1500, 2000]
cub_n_prot = [0, 0, 0, 0, 0, 0, 1, 6, 24, 58, 120, 203, 298, 342, 528, 746, 990, 1420]
cub_acc = [0.818, 0.817, 0.830, 0.846, 0.845, 0.867, 0.869, 0.870, 0.869, 0.861, 0.861, 0.868, 0.866, 0.871, 0.881, 0.876, 0.88, 0.871]

# Plotting empty prototypes
plt.figure(figsize=(10, 6))
plt.plot(cub_n, cub_n_prot, marker='o', linestyle='-', color='steelblue')

# Labels and Title
plt.xlabel('Number of prototypes', fontsize=20)
plt.ylabel('Number of empty prototypes', fontsize=20)
plt.grid(True)

# Making the ticks bigger
plt.tick_params(axis='both', which='major', labelsize=18)

plt.savefig(f"plots/plot_cub_n.png")
plt.close()

# Plotting accuracy
plt.figure(figsize=(10, 6))
plt.plot(cub_n, cub_acc, marker='o', linestyle='-', color='firebrick')

# Labels and Title
plt.xlabel('Number of prototypes', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.grid(True)

# Making the ticks bigger
plt.tick_params(axis='both', which='major', labelsize=18)

plt.savefig(f"plots/plot_cub_acc.png")


# AwA2 plots ------------------------------------------------------------------------------------------------------------------------------------
awa_n = [25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 750, 800, 850, 1000]
awa_acc = [0.803, 0.842, 0.873, 0.876, 0.884, 0.887, 0.882, 0.883, 0.880, 0.888, 0.896, 0.902, 0.895, 0.894, 0.904, 0.901, 0.894]
awa_n_prot = [0,0,0,0,0,0,1,6,9,32,72,139,177,221,234,284,378]

# Plotting empty prototypes
plt.figure(figsize=(10, 6))
plt.plot(awa_n, awa_n_prot, marker='o', linestyle='-', color='steelblue')

# Labels and Title
plt.xlabel('Number of prototypes', fontsize=20)
plt.ylabel('Number of empty prototypes', fontsize=20)
plt.grid(True)

# Making the ticks bigger
plt.tick_params(axis='both', which='major', labelsize=18)

plt.savefig(f"plots/plot_awa_n.png")
plt.close()

# Plotting accuracy
plt.figure(figsize=(10, 6))
plt.plot(awa_n, awa_acc, marker='o', linestyle='-', color='firebrick')

# Labels and Title
plt.xlabel('Number of prototypes', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.grid(True)

# Making the ticks bigger
plt.tick_params(axis='both', which='major', labelsize=18)

plt.savefig(f"plots/plot_awa_acc.png")
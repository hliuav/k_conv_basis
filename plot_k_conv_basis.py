#import torch
#import matplotlib.pyplot as plt
#import numpy as np
#
## Initialize lists to store the data
#flops_conv = []
#relative_diff = []
#
## Read data from files
#ks = [5, 10, 20, 40, 80, 160, 320, 640] #, 1280, 2560, 5120, 10240]
#for k in ks:  # from 1 to 10
#    # Load flops data
#    flops_path = f'flops_approx_{k}.pth'
#    flops_data = torch.load(flops_path)
#    flops_conv.append(flops_data)
#    
#    # Load relative difference data
#    diff_path = f'relative_diff_{k}.pth'
#    diff_data = torch.load(diff_path)
#    relative_diff.append(diff_data)
#
## Convert lists to numpy arrays for easier handling in plotting
#flops_conv = np.array(flops_conv)
#relative_diff = np.array(relative_diff)
#
## Create a figure and a set of subplots
#fig, ax1 = plt.subplots()
#
## Plot flops_conv on ax1
#color = 'tab:red'
#ax1.set_xlabel('k conv basis')
#ax1.set_ylabel('FLOPS Approx', color=color)
##ax1.plot(ks, flops_conv, color=color)
#ax1.semilogy(ks, flops_conv, color=color, marker='o', markersize=10)  # Use semilogy for log scale on y-axis
#ax1.tick_params(axis='y', labelcolor=color)
#ax1.set_xscale('log')  # Log scale for x-axis
#
#flops_naive_value = 4142992032015
#ax1.axhline(y=flops_naive_value, color='green', linestyle='--', linewidth=2, label='FLOPS Naive')
#
## Create ax2 for the relative difference with the same x-axis
#ax2 = ax1.twinx()
#color = 'tab:blue'
#ax2.set_ylabel('Relative Difference', color=color)
#ax2.plot(ks, relative_diff, color=color, marker='s', markersize=10)
#ax2.tick_params(axis='y', labelcolor=color)
#
## Title and show the plot
#plt.title('FLOPS and Relative Differences for k Conv Basis')
#fig.legend()  # Added legend
#plt.show()
import torch
import matplotlib.pyplot as plt
import numpy as np

ks = [5, 10, 20, 40, 80, 160, 320, 640] #, 1280, 2560, 5120, 10240]
# Initialize lists to store the data
flops_approx_sets = [[] for _ in range(3)]
relative_diff_sets = [[] for _ in range(3)]

# Read data from files
for k in ks:  # from 1 to 10
    for i in range(3):  # for each of the three sets of data
        # Load flops data
        flops_path = f'flops_approx_{k}_{i}.pth'
        flops_data = torch.load(flops_path)
        flops_approx_sets[i].append(flops_data)
        
        # Load relative difference data
        diff_path = f'relative_diff_{k}_{i}.pth'
        diff_data = torch.load(diff_path)
        relative_diff_sets[i].append(diff_data)

# Convert lists to numpy arrays for easier handling in calculations
flops_approx_sets = [np.array(approx) for approx in flops_approx_sets]
relative_diff_sets = [np.array(diff) for diff in relative_diff_sets]

# Calculate the means and standard deviations
flops_approx_mean = np.mean(flops_approx_sets, axis=0)
flops_approx_std = np.std(flops_approx_sets, axis=0)
relative_diff_mean = np.mean(relative_diff_sets, axis=0)
relative_diff_std = np.std(relative_diff_sets, axis=0)

flops_approx_mean = flops_approx_mean.reshape(-1)
relative_diff_mean = relative_diff_mean.reshape(-1)
flops_approx_std = flops_approx_std.reshape(-1)
relative_diff_std = relative_diff_std.reshape(-1)
print(flops_approx_mean.shape)
print(flops_approx_std.shape)

# Create a figure and a set of subplots
plt.rcParams.update({'font.size': 30, 'legend.fontsize': 30})
fig, ax1 = plt.subplots()

# Plot flops_approx_mean on ax1 with log scale for y and error bars
# Plot flops_approx_mean on ax1 with log scale for y and error bars
color = 'tab:red'
ax1.set_xlabel('k')
ax1.set_ylabel('FLOPs Approx', color=color)
ax1.errorbar(ks, flops_approx_mean, yerr=flops_approx_std, fmt='-o', color=color, markersize=5, label='FLOPs Approx')  
ax1.fill_between(ks, flops_approx_mean - flops_approx_std, flops_approx_mean + flops_approx_std, color=color, alpha=0.2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xscale('log')  # Log scale for x-axis
ax1.set_yscale('log')  # Log scale for y-axis

# Add the FLOPS Naive baseline
flops_naive_value = 4142992032015
ax1.axhline(y=flops_naive_value, color='purple', linestyle='--', linewidth=2, label='FLOPs Naive')

# Create ax2 for the relative difference with the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Relative Difference', color=color)
#ax2.errorbar(ks, relative_diff_mean, yerr=relative_diff_std, fmt='-s', color=color, markersize=5, label='Relative Difference')
ax2.plot(ks, relative_diff_mean, '-s', color=color, markersize=5, label='Relative Difference')
ax2.errorbar(ks, relative_diff_mean, yerr=relative_diff_std, fmt='none', ecolor=color, capsize=5)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(bottom=0.000, top=0.008)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# Title and show the plot with legend
plt.title('FLOPs and Relative Diff')
fig.legend(loc="lower right", bbox_to_anchor=(0.9, 0.15))
plt.gcf().set_size_inches(13, 8.5)
plt.tight_layout()
plt.savefig("flops_and_relative_diff.pdf")
#plt.show()


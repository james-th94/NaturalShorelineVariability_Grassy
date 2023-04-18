# %% Code Description
"""
Code to create a subplot from images.
Created 18/04/23, originally for Coast and Ports conference figures.
"""

# %% Packages and Directory
projDir = "c:\\Users\\s5245653\\OneDrive - Griffith University\\Projects\\NaturalShorelineVariability_Grassy\\"
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import string

os.chdir(projDir)
# %% User Inputs
images = [
    projDir + "data\\Plots\\Results\\New\\WestEnd.png",
    projDir + "data\\Plots\\Results\\New\\CentralBeach.png",
    projDir + "data\\Plots\\Results\\New\\EastEnd.png",
]

# Plot configurations
m = 3  # Number of rows
n = 1  # Number of columns
outputFig = projDir + "data\\Plots\\Results\\New\\TotalBeach.png"
DPI = 600
horizontalPadding = 0.1
verticalPadding = 0.1


# %% Plotting

if m > 1 and n > 1:
    fig, axes = plt.subplots(m, n)
    for i in range(m):
        for j in range(n):
            img = mpimg.imread(images[(i + 1) * j])
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            axes[i, j].text(
                0.1,
                0.8,
                "(" + string.ascii_lowercase[(i + 1) * j] + ")",
                transform=axes[i, j].transAxes,
                size=12,
            )
    plt.subplots_adjust(hspace=horizontalPadding, wspace=verticalPadding)
    plt.savefig(outputFig, bbox_inches="tight", dpi=DPI)

elif m > 1 and n == 1:
    fig, axes = plt.subplots(m)
    for i in range(m):
        img = mpimg.imread(images[i])
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].text(
            0.05,
            0.1,
            "(" + string.ascii_lowercase[i] + ")",
            transform=axes[i].transAxes,
            size=10,
            color="k",
        )
    plt.subplots_adjust(hspace=horizontalPadding, wspace=verticalPadding)
    plt.savefig(outputFig, bbox_inches="tight", dpi=DPI)

elif n > 1 and m == 1:
    fig, axes = plt.subplots(n)
    for i in range(n):
        img = mpimg.imread(images[i])
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].text(
            0.1,
            0.8,
            "(" + string.ascii_lowercase[i] + ")",
            transform=axes[i].transAxes,
            size=12,
        )
    plt.subplots_adjust(hspace=horizontalPadding, wspace=verticalPadding)
    plt.savefig(outputFig, bbox_inches="tight", dpi=DPI)

else:
    print("Error in number of cols and/or number of rows (m and n values).")

# %%

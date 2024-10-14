import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_style():
    # Set the style to a more modern look
    sns.set_style("whitegrid", {"grid.linestyle": "--"})

    # Set a custom color palette with brighter colors
    bright_palette = [
        "#FF9999",
        "#66B2FF",
        "#99FF99",
        "#FFCC99",
        "#FF99CC",
        "#FFD700",
        "#00CED1",
        "#FF6347",
        "#40E0D0",
        "#FF1493",
    ]
    sns.set_palette(bright_palette)

    # Set figure size
    plt.rcParams["figure.figsize"] = (12, 7)

    # Set font styles
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 14

    # Line styles and widths
    plt.rcParams["lines.linewidth"] = 2.5
    plt.rcParams["lines.solid_capstyle"] = "round"

    # Grid and ticks
    plt.rcParams["grid.color"] = "#E0E0E0"
    plt.rcParams["grid.alpha"] = 0.8
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.major.size"] = 5
    plt.rcParams["ytick.major.size"] = 5
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True

    # Axis properties
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 1 
    plt.rcParams["axes.edgecolor"] = "#212121"  

    # Legend
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.loc"] = "best"

    # Set the color cycle to use our bright palette
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=bright_palette)

    # Display settings for notebook or scripts
    plt.rcParams["figure.autolayout"] = True


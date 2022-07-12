import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np

## TAILLE DES FIGURES

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    #golden_ratio = .9
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    #fig_height_in = fig_width_in * .9 * (subplots[0] / subplots[1])
    tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 14,
    "font.size": 14,
    # Make the legend/label fonts a little smaller
    "legend.fontsize":11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
    }

    plt.rcParams.update(tex_fonts)
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{lmodern}', #lmodern: lateX font; tgheros: helvetica font
    r'\usepackage{sansmath}', # math-font matching helvetica
    r'\sansmath' # actually tell tex to use it!
    r'\usepackage[scientific-notation=false]{siunitx}', # micro symbols
    r'\sisetup{detect-all}', # force siunitx to use the fonts
    ]
    
    return (fig_width_in, fig_height_in)

    ## FONTS


## CHOIX DES COULEURS

def vcolors(n=7):
        return plt.cm.viridis(np.linspace(0,1,n))
import seaborn as sns
from IPython.core.display import HTML

def format_dataframes():
    css = open("./css/style-table.css").read()
    return HTML("<style>{}</style>".format(css))

def format_plots(fig_scale=10., colorblind=False):
    if colorblind:
        palette = "colorblind"
    else:
        palette = "deep"
    sns.set(color_codes=True, palette=palette,
        font_scale=1.5, context="notebook",
        rc = {'figure.figsize':[fig_scale, fig_scale]})
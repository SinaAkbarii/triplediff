import numpy as np
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# read
sample_sizes = list(range(1000, 11000, 1000))
with open('results/synthetic/results_rc.pkl', 'rb') as f:
    results = pickle.load(f)



# Custom palette
palette = {
    'correct': '#4E79A7',
    'or_misspec': '#F4CC4A',#'#AF7AA1',
    'ps_misspec': '#59A14F',
    'both_misspec': '#E15759'
}


# box plot results[sample_size]['correct']['rel_bias'] for sample_size in sample_sizes

def plot_results(results, sample_sizes, width=1, dodge_width=0.8,name='bias'):
    # Prepare long-form DataFrame
    records = []
    settings = ['correct', 'or_misspec', 'ps_misspec', 'both_misspec']

    for sample_size in sample_sizes:
        for setting in settings:
            biases = results[sample_size][setting]['rel_bias']
            for bias in biases:
                records.append({
                    'Sample Size': sample_size,
                    'Setting': setting,
                    'Relative Bias': bias
                })

    df = pd.DataFrame.from_records(records)

    # Set Seaborn theme
    sns.set_theme(style='whitegrid', font_scale=1.2, rc={
        'axes.edgecolor': 'gray',
        'grid.color': '#e5e5e5',
        'axes.spines.right': False,
        'axes.spines.top': False,
    })

    textwidth_in = 146.8 / 25.4
    fig, ax = plt.subplots(1, 1, figsize=(textwidth_in, textwidth_in * 0.7))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["utopia"],  # Fourier uses Utopia font
        "font.size": 11,  # match LaTeX documentclass 11pt
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.latex.preamble": r"\usepackage{fourier}",  # Optional but makes it match even closer
        "axes.formatter.useoffset": True,
        "axes.formatter.use_mathtext": True,
    })
    sns.boxplot(
        data=df,
        x='Sample Size',
        y='Relative Bias',
        hue='Setting',
        palette=palette,
        linewidth=0.6,
        fliersize=1.5,
        ax=ax,
        # make the boxes thinner
        width=width,
        # add a bit of space between boxes of different settings
        dodge=True
    )
    # Overlay mean values
    mean_df = df.groupby(['Sample Size', 'Setting'])['Relative Bias'].mean().reset_index()

    # Unique values in plotting order
    sample_size_order = sorted(df['Sample Size'].unique())
    setting_order = ['correct', 'or_misspec', 'ps_misspec', 'both_misspec']  # ensure consistent hue order
    # setting_order = ['correctly specified', 'OR misspecified', 'PS misspecified',
    #                  'both misspecified']  # ensure consistent hue order

    # Mapping from sample size and setting to a x-position (with dodge)
    n_settings = len(setting_order)
    # dodge_width = 0.7  # total space allocated to group
    box_width = dodge_width / n_settings

    # Offset map: gives center of each subgroup
    offsets = {setting: (-dodge_width / 2 + box_width / 2 + i * box_width) for i, setting in enumerate(setting_order)}
    ss_to_idx = {ss: i for i, ss in enumerate(sample_size_order)}

    handles, labels = ax.get_legend_handles_labels()

    # Original order is ['correct', 'or_misspec', 'ps_misspec', 'both_misspec'] + possibly other labels
    label_map = {
        'correct': 'Correctly specified',
        'or_misspec': 'OR misspecified',
        'ps_misspec': 'PS misspecified',
        'both_misspec': 'Both misspecified'
    }

    new_labels = [label_map.get(lbl, lbl) for lbl in labels]

    ax.legend(handles, new_labels, loc='lower right', ncol=2, fontsize=11)
    # ax.legend(loc='lower right', ncol=2, fontsize=11)# title='Model Setting', title_fontsize=12, loc='lower right')
    # Plot means
    for setting in setting_order:
        subset = mean_df[mean_df['Setting'] == setting]
        xs = [ss_to_idx[ss] + offsets[setting] for ss in subset['Sample Size']]
        ys = subset['Relative Bias']

        ax.scatter(
            x=xs,
            y=ys,
            color=palette[setting],
            marker='D',
            edgecolor='black',
            zorder=5,
            s=4,
            linewidth=0.5,
            label=f'{setting} mean'  # optional: may remove if legend duplicates
        )

    # Update x-axis ticks and labels:
    sample_size_order = sorted(df['Sample Size'].unique())
    # Positions are integers from 0 to len(sample_size_order)-1 by default in seaborn boxplot, so set xticks accordingly
    ax.set_xticks(range(len(sample_size_order)))
    # Set labels divided by 1000
    ax.set_xticklabels([str(int(ss / 1000)) for ss in sample_size_order])
    # Add multiplier text on the axis
    ax.annotate(
        r"$\times 10^3$",
        xy=(0.95, -0.08),  # original position (right bottom outside the axis)
        xycoords='axes fraction',
        ha='left',
        va='top',
        fontsize=9,  # smaller font
        xytext=(-10, 0),  # shift left by 10 points (negative x)
        textcoords='offset points'
    )
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    # Prettify the axes
    # ax.set_title('Relative Bias by Sample Size and Model Setting', fontsize=16, fontweight='bold')
    ax.set_xlabel('Sample Size', fontsize=11)
    ax.set_ylabel('Relative Bias', fontsize=11)
    if name == 'bias_p':
        plt.ylim(-1.26, .66)
    elif name == 'bias_rc':
        pass
    plt.tight_layout()
    # save with high quality
    plt.savefig('results/synthetic/' + name + '.pdf', bbox_inches='tight', dpi=300)
    plt.show()


plot_results(results, sample_sizes, .6, .59, 'bias_rc')




























# line plot results[sample_size][model]['covered'] vs sample_size for every model in ['correct', 'or_misspec', 'ps_misspec', 'both_misspec']
def plot_coverage(results, sample_sizes):
    # Prepare long-form DataFrame
    records = []
    models = ['correct', 'or_misspec', 'ps_misspec', 'both_misspec']

    for sample_size in sample_sizes:
        for model in models:
            coverage = np.array(results[sample_size][model]['covered']).mean()
            records.append({
                'Sample Size': sample_size,
                'Model': model,
                'Coverage': coverage
            })

    df = pd.DataFrame.from_records(records)

    # Set Seaborn theme
    sns.set_theme(style='whitegrid', font_scale=1.2, rc={
        'axes.edgecolor': 'gray',
        'grid.color': '#e5e5e5',
        'axes.spines.right': False,
        'axes.spines.top': False,
    })

    # Custom palette
    palette = {
        'correct': '#4E79A7',
        'or_misspec': '#AF7AA1',
        'ps_misspec': '#59A14F',
        'both_misspec': '#E15759'
    }

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    sns.lineplot(
        data=df,
        x='Sample Size',
        y='Coverage',
        hue='Model',
        palette=palette,
        marker='o',
        linewidth=2.5,
        ax=ax
    )

    ax.legend(title='Model Setting', fontsize=11, title_fontsize=12, loc='lower right')


    # Prettify the axes
    ax.set_title('Coverage by Sample Size and Model Setting', fontsize=16, fontweight='bold')
    ax.set_xlabel('Sample Size', fontsize=13)
    ax.set_ylabel('Coverage', fontsize=13)
    plt.ylim(0.8, 1.0)
    plt.tight_layout()
    plt.show()
plot_coverage(results, sample_sizes)


# histogram of results[10000][model]['rel_bias']  for model in ['correct', 'or_misspec', 'ps_misspec', 'both_misspec']
def plot_histograms(results, sample_sizes):
    # Prepare long-form DataFrame
    records = []
    models = ['correct', 'or_misspec', 'ps_misspec', 'both_misspec']

    for sample_size in sample_sizes:
        for model in models:
            biases = results[sample_size][model]['rel_bias']
            for bias in biases:
                records.append({
                    'Sample Size': sample_size,
                    'Model': model,
                    'Relative Bias': bias
                })

    df = pd.DataFrame.from_records(records)

    # Set Seaborn theme
    sns.set_theme(style='whitegrid', font_scale=1.2, rc={
        'axes.edgecolor': 'gray',
        'grid.color': '#e5e5e5',
        'axes.spines.right': False,
        'axes.spines.top': False,
    })

    g = sns.FacetGrid(df, col='Model', hue='Model', palette=palette, col_wrap=2, height=5)
    g.map(sns.histplot, 'Relative Bias', kde=True, bins=30, stat='density')

    g.add_legend(title='Model Setting')

    # Prettify the axes
    g.set_titles(col_template='{col_name}')
    g.set_axis_labels('Relative Bias', 'Density')

    plt.tight_layout()
    plt.show()

plot_histograms(results, sample_sizes)








def plot_caterpillar(results, sample_size, name='caterpillar', draw_legend=True):
    true_att = 10
    models = ['correct', 'or_misspec', 'ps_misspec', 'both_misspec']
    label_map = {
        'correct': 'Correctly specified',
        'or_misspec': 'OR misspecified',
        'ps_misspec': 'PS misspecified',
        'both_misspec': 'Both misspecified'
    }

    # Match LaTeX figure width and style from plot_results
    textwidth_in = 146.8 / 25.4 /2  # convert mm to inches
    fig, ax = plt.subplots(1, 1, figsize=(textwidth_in, textwidth_in * 0.7))

    # Seaborn theme & LaTeX text settings
    sns.set_theme(style='whitegrid', font_scale=1.2, rc={
        'axes.edgecolor': 'gray',
        'grid.color': '#e5e5e5',
        'axes.spines.right': False,
        'axes.spines.top': False,
    })
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["utopia"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.latex.preamble": r"\usepackage{fourier}",
        "axes.formatter.useoffset": True,
        "axes.formatter.use_mathtext": True,
    })

    # Caterpillar plot
    for i, model in enumerate(reversed(models)):
        ests = np.array(results[sample_size][model]['estimated_atts'])
        ci_lengths = np.array(results[sample_size][model]['ci_lengths'])

        ci_lowers = ests - ci_lengths / 2
        ci_uppers = ests + ci_lengths / 2
        includes_true = (ci_lowers <= true_att) & (ci_uppers >= true_att)

        # Jitter Y positions
        y_vals = np.random.uniform(i - 0.4, i + 0.4, size=len(ests))

        for est, lower, upper, y, covers in zip(ests, ci_lowers, ci_uppers, y_vals, includes_true):
            ax.plot(
                [lower, upper], [y, y],
                color=palette[model],
                alpha=0.3 if covers else 0.1,
                linewidth=1
            )

    # True value line
    ax.axvline(true_att, color='black', linestyle='--', linewidth=1.5)

    # Y-axis ticks and labels
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([label_map[m] for m in reversed(models)])

    # Labels and limits
    # ax.set_xlabel('Estimate with 95\% CI')
    # ax.set_xlim(true_att - 15, true_att + 5)

    # Legend
    if draw_legend:
        # Create custom legend handles
        handles = [plt.Line2D([0], [0], color=palette[m], lw=2) for m in models]
        ax.legend(handles, [label_map[m] for m in models],
                  loc='upper left', ncol=1, fontsize=10)

    # Remove y-axis line and ticks
    ax.spines[['left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(f'results/synthetic/{name}.pdf', bbox_inches='tight', dpi=300)
    plt.show()
plot_caterpillar(results, 10000, 'cat_rc', False)


## print coverage:
model = 'correct'
coverage = [100*np.mean(results[sample_size][model]['covered']) for sample_size in sample_sizes]
# print coverage values separated with &
print(' & '.join([f'{cov:.1f}' for cov in coverage]))


model = 'correct'

## print bias:
bias = [np.mean(results[sample_size][model]['bias']) for sample_size in sample_sizes]
# print relative bias values separated with &
print(' & '.join([f'{rb:.3f}' for rb in bias]))

## print RMSE:
# mse is the ['bias']^2 + ['variance']/num_samples
model = 'correct'
rmse = [np.sqrt(np.mean(
    np.array(results[sample_size][model]['bias'])**2)
) for sample_size in sample_sizes]
# print MSE values separated with &
print(' & '.join([f'{rmse_val:.3f}' for rmse_val in rmse]))

with open('results/empirical.pkl', 'rb') as f:
    resemp = pickle.load(f)

# histogram of resemp['estimated_atts']
plt.hist(resemp['estimated_atts'], bins=100)
plt.show()
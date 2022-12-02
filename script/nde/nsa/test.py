import matplotlib.lines as mlines
k_line = mlines.Line2D([], [], lw=6, color='dimgray',
                       label='GAMA DR3 spec-$z$ and stellar mass')
b_line = mlines.Line2D([], [], color='tomato', lw=6,
                       label=r'Inferred galaxy population based on GAMA DR3 photometry')

labels = [
    r'$\beta_{1}$',
    r'$\beta_{2}$',
    r'$\beta_{3}$',
    r'$\beta_{4}$',
    r'$f_{\mathrm{burst}}$',
    r'$t_{\mathrm{burst}}$',
    r'$\log\,(Z_\star/Z_\odot)$',
    r'$\tau_1$',
    r'$\tau_2$',
    r'$n_{\mathrm{dust}}$',
    r'$z$',
    r'$\log\,M_{\star}$',
]

if name == 'NMF_ZH':
    labels = labels[:6] + [r'$\gamma_{1}$', r'$\gamma_{2}$'] + labels[7:]


y_truth_trans = np.hstack([_samples[:, 1:],  # params taken by emulator, including redshift (for t_age)
                           _samples[:, 0:1],  # stellar mass
                           ])
K = y_truth_trans.shape[1]
figure, axes = plt.subplots(K, K, figsize=(25, 25))

ext_axes = np.array([[figure.get_axes()[12 * 11 - 2], figure.get_axes()[12 * 11 - 1]],
                     [figure.get_axes()[12 * 12 - 2], figure.get_axes()[12 * 12 - 1]]])

figure = corner.corner(
    np.hstack([z_mass_met[:, 0:1], z_mass_met[:, 1:2] + 0.3]),
    fig=figure,
    ext_axes=ext_axes,
    color='dimgray',
    labels=None,  # ['$z$', '$\log\ M_\star$', '$\log\ Z$'],
    range=[[0, 0.7], [7.7, 13]],
    bins=25,
    smooth=1.0,
    fill_contours=True,
    show_titles=False,
    title_kwargs={'fontsize': 22, 'loc': 'left'},
    label_kwargs={'fontsize': 32},
    labelpad=0.15,
    hist_kwargs={'density': True, 'lw': 1},
    plot_datapoints=False
)
ext_axes[1, 0].set_yticks([])


figure = corner.corner(
    y_truth_trans,
    fig=figure,
    color='#cf484b',
    yfactor=1.3,
    labels=labels,
    range=[[0, 1], [0, 1], [0., 1], [0, 1], [-0.02, 1.02], [0.5, 13],
           [-2.6, 0.35], [0, 3], [0, 2.5], [-1.5, 1], [0, 0.7], [7.7, 13]],
    bins=25,
    smooth=1.0,
    fill_contours=True,
    show_titles=True,
    title_kwargs={'fontsize': 25, 'loc': 'left'},
    label_kwargs={'fontsize': 34},
    labelpad=0.15,
    hist_kwargs={'density': True, 'lw': 1},
    plot_datapoints=False
)


for ax in figure.get_axes():
    ax.tick_params(labelsize=21, length=4)
    ax.minorticks_off()

ax = figure.get_axes()[10]
leg = ax.legend(handles=[b_line, k_line],
                fontsize=34,
                bbox_to_anchor=(0., 1.0, 1.3, .0))

plt.suptitle(
    r'GAMA DR3 Aperture Matched Photometry $\texttt{AUTO}$, SNR $>$ 1', fontsize=40, y=1.02)
plt.savefig('/tigress/jiaxuanl/public_html/figure/popsed/gama_dr3_snr1_posterior.pdf',
            bbox_inches='tight')

figure = corner.corner(np.vstack([obs_truth['logmstar'], obs_truth['redshift'],
                                  np.log10(obs_truth['sfr']), obs_truth['logzsol'], obs_truth['age']]).T,
                       labels=['logmstar', 'redshift', 'SFR', 'Z', 'Age'], bins=40,
                       color='gray',  # quantiles=[0.16, 0.5, 0.84],
                       smooth=0.8, fill_contours=True,
                       show_titles=True, title_kwargs={"fontsize": 14},
                       hist_kwargs={'density': True}, plot_datapoints=True)

figure = corner.corner(np.vstack([obs_rec['logmstar'], obs_rec['redshift'],
                                  np.log10(obs_rec['sfr']), obs_rec['logzsol'], obs_rec['age']]).T,
                       fig=figure,
                       labels=['logmstar', 'redshift', 'SFR', 'Z', 'Age'], bins=40,
                       color='dodgerblue',  # quantiles=[0.16, 0.5, 0.84],
                       smooth=0.8, fill_contours=True,
                       show_titles=True, title_kwargs={"fontsize": 14},
                       hist_kwargs={'density': True}, plot_datapoints=True)

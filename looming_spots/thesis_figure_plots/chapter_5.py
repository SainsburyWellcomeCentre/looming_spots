from looming_spots.thesis_figure_plots import photometry_example_traces
import itertools
import pingouin as pg


def compute_stats_high_contrast_photometry_d1_d2_snl():

    df = photometry_example_traces.get_signal_df(
        groups=('d1flexGCaMP_var_contrast', 'd2flexGCaMP_var_contrast', 'photometry_habituation_tre-GCaMP-contrasts'))
    df = df[df['loom number'] < 12]
    df = df[df['contrast'] == 0]

    groups = ('d1flexGCaMP_var_contrast', 'd2flexGCaMP_var_contrast', 'photometry_habituation_tre-GCaMP-contrasts')
    folder = '/home/slenzi/looming/dataframes/loom_number_snl_d1_d2_photometry_high_contrast'
    for group in groups:
        sub_df = df[df['group'].isin([group,])]
        sub_df['loom number'] = sub_df['loom number'].astype(str)
        aov = pg.anova(
            dv="deltaf metric",
            between="loom number",
            data=sub_df,
        )
        sub_df = df[df['group'].isin([group, ])]
        sub_df['loom number'] = sub_df['loom number'].astype(str)
        summary_table = pg.pairwise_ttests(
            dv="deltaf metric",
            within="loom number",
            subject="mouse id",
            data=sub_df,
        )
        aov.to_csv(f'{folder}/{group}_within_first_loom_4stim.csv')
        summary_table.to_csv(f'{folder}/{group}_post_hocs_first_loom_4stim.csv')

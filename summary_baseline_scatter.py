import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotting_functions as pf
import utils


plt.rcParams['font.family'] = 'Arial'


baseline_export_folder_name = 'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS baselines CL-Ali\\'
summary_output_folder = os.path.join(baseline_export_folder_name, 'summary scatterplots')
excluded_animals = ['VIV49933']
compute_statistics = 1
statistics_test = 'nonparametric'
uniform_ranges = 1

param_display_labels = {
    'COO': 'COO',
    'SL': 'SL',
    'DS': 'DS',
    'CS': 'CS',
    'SW': 'SW',
    'phase': 'phase',
    'SS': 'SS',
}
param_order = ['COO', 'SL', 'DS', 'CS', 'SW', 'phase', 'SS']
cohort_order = ['WT', 'Linj', 'Rinj', 'RLinj']
cohort_display_names = ['non-inj', 'Linj', 'Rinj', 'RLinj']
bars_ranges = {
    'COO': [-2, 4],
    'SL': [-5, 9],
    'DS': [-12, 5],
    'CS': [-5, 5],
    'SW': [-5, 12],
    'phase': [-4, 7],
    'SS': [-0.4, -0.2],
}

lightseagreen_light = pf.lighten_color('lightseagreen', 0.55)
lightseagreen_dark = pf.darken_color('lightseagreen', 0.4)
cohort_colors = {
    'WT': 'gray',
    'Linj': lightseagreen_dark,
    'Rinj': lightseagreen_light,
    'RLinj': 'darkblue',
}


def parse_baseline_filename(filename):
    file_root, file_ext = os.path.splitext(filename)
    if file_ext.lower() != '.csv' or '_' not in file_root:
        return None

    reference_suffix = ''
    if file_root.endswith('_ipsi_hind_ref'):
        file_root = file_root[:-len('_ipsi_hind_ref')]
        reference_suffix = 'ipsi_hind_ref'

    param_label, tag = file_root.split('_', 1)
    cohort = None
    for cohort_name in ['WT', 'LATinj', 'Linj', 'Rinj', 'RLinj']:
        if tag == cohort_name:
            cohort = cohort_name
            session_tag = ''
            break
        if tag.startswith(cohort_name + '_'):
            cohort = cohort_name
            session_tag = tag[len(cohort_name) + 1:]
            break

    if cohort is None:
        return None

    return {
        'param_label': param_label,
        'cohort': cohort,
        'session_tag': session_tag,
        'reference_suffix': reference_suffix,
    }


def get_plot_key(param_label, reference_suffix):
    if not reference_suffix:
        return param_label
    return f'{param_label}_{reference_suffix}'


def get_plot_filename(plot_key):
    return f'{plot_key}_summary_scatter'


def get_plot_ylabel(param_label, reference_suffix):
    ylabel = param_display_labels.get(param_label, param_label)
    if reference_suffix == 'ipsi_hind_ref':
        ylabel += ' baseline (ipsi hind ref)'
    else:
        ylabel += ' baseline'
    return ylabel


def get_session_bucket(cohort, session_tag):
    session_tag = session_tag.lower()
    if cohort == 'WT':
        return 'single'

    if cohort == 'RLinj':
        if 'contra-fast' in session_tag or 'right-fast' in session_tag:
            return 'session_a'
        if 'ipsi-fast' in session_tag or 'left-fast' in session_tag:
            return 'session_b'
        return 'single'

    if 'left-fast' in session_tag or 'contra-fast' in session_tag:
        return 'session_a'
    if 'right-fast' in session_tag or 'ipsi-fast' in session_tag:
        return 'session_b'
    return 'single'


def load_baseline_data(folder_path):
    grouped_data = {}
    for filename in os.listdir(folder_path):
        parsed_filename = parse_baseline_filename(filename)
        if parsed_filename is None:
            continue
        if parsed_filename['cohort'] == 'LATinj':
            continue
        if parsed_filename['param_label'] not in param_display_labels:
            continue

        plot_key = get_plot_key(parsed_filename['param_label'], parsed_filename['reference_suffix'])
        grouped_data.setdefault(plot_key, {
            'param_label': parsed_filename['param_label'],
            'reference_suffix': parsed_filename['reference_suffix'],
            'cohorts': {cohort: {} for cohort in cohort_order},
        })

        data_frame = pd.read_csv(os.path.join(folder_path, filename))
        if 'animal' not in data_frame.columns or 'baseline_value' not in data_frame.columns:
            continue

        cohort_data = grouped_data[plot_key]['cohorts'][parsed_filename['cohort']]
        session_bucket = get_session_bucket(parsed_filename['cohort'], parsed_filename['session_tag'])

        for row in data_frame.itertuples(index=False):
            if pd.isna(row.animal) or pd.isna(row.baseline_value):
                continue
            animal_name = str(row.animal)
            if animal_name in excluded_animals:
                continue
            baseline_value = float(row.baseline_value)
            cohort_data.setdefault(animal_name, {}).setdefault(session_bucket, []).append(baseline_value)

    return grouped_data


def average_animal_sessions(animal_sessions):
    averaged_values = []
    animal_names = []
    for animal_name in sorted(animal_sessions):
        bucket_means = []
        for bucket_values in animal_sessions[animal_name].values():
            if len(bucket_values) == 0:
                continue
            bucket_means.append(np.nanmean(np.asarray(bucket_values, dtype=float)))
        if len(bucket_means) == 0:
            continue
        averaged_value = np.nanmean(np.asarray(bucket_means, dtype=float))
        if np.isfinite(averaged_value):
            animal_names.append(animal_name)
            averaged_values.append(averaged_value)
    return animal_names, np.asarray(averaged_values, dtype=float)


def build_plot_arrays(grouped_data):
    plot_arrays = {}
    for plot_key, plot_info in grouped_data.items():
        cohort_raw_values = []
        cohort_animals = {}
        for cohort_name in cohort_order:
            animals, values = average_animal_sessions(plot_info['cohorts'][cohort_name])
            cohort_animals[cohort_name] = animals
            cohort_raw_values.append(values)

        max_animals = max((len(values) for values in cohort_raw_values), default=0)
        cohort_values = []
        point_colors = []
        pval_vs_zero = []
        for cohort_name, values in zip(cohort_order, cohort_raw_values):
            padded_values = np.full(max_animals, np.nan)
            padded_values[:len(values)] = values
            cohort_values.append(padded_values)
            point_colors.append([cohort_colors[cohort_name]] * len(padded_values) if len(padded_values) > 0 else None)
            if compute_statistics:
                pvalue, _ = utils.compare_metric_against_zero(values, statistics_test)
                pval_vs_zero.append(pvalue)

        plot_arrays[plot_key] = {
            'param_label': plot_info['param_label'],
            'reference_suffix': plot_info['reference_suffix'],
            'cohort_values': cohort_values,
            'cohort_animals': cohort_animals,
            'point_colors': point_colors,
            'pval_vs_zero': pval_vs_zero,
        }
    return plot_arrays


def get_plot_ylim(param_label):
    if not uniform_ranges or param_label not in bars_ranges:
        return None
    max_abs_limit = max(abs(value) for value in bars_ranges[param_label])
    scaled_limit = 1.5 * max_abs_limit
    return [-scaled_limit, scaled_limit]


def plot_summary_scatter(plot_info):
    return pf.plot_symmetry_scatterplot(
        plot_info['cohort_values'],
        cohort_order,
        [cohort_colors[cohort_name] for cohort_name in cohort_order],
        get_plot_ylabel(plot_info['param_label'], plot_info['reference_suffix']),
        stat_results=plot_info['pval_vs_zero'] if compute_statistics else None,
        ylim=get_plot_ylim(plot_info['param_label']),
        point_colors=plot_info['point_colors'],
        display_names=cohort_display_names,
        stat_mode='vs_zero',
        connect_points=False,
    )


def print_summary_counts(plot_key, cohort_animals):
    counts = ', '.join(f'{cohort}={len(cohort_animals[cohort])}' for cohort in cohort_order)
    print(f'{plot_key}: {counts}')


def print_summary_stats(plot_key, pval_vs_zero):
    if not compute_statistics:
        return
    stats_values = []
    for cohort_name, pvalue in zip(cohort_order, pval_vs_zero):
        if np.isnan(pvalue):
            stats_values.append(f'{cohort_name}=nan')
        else:
            stats_values.append(f'{cohort_name}={pvalue:.6f}')
    print(f'{plot_key} p-values vs 0: ' + ', '.join(stats_values))


def main():
    if not os.path.exists(baseline_export_folder_name):
        raise FileNotFoundError(f'Baseline export folder not found: {baseline_export_folder_name}')

    grouped_data = load_baseline_data(baseline_export_folder_name)
    plot_arrays = build_plot_arrays(grouped_data)

    if not os.path.exists(summary_output_folder):
        os.mkdir(summary_output_folder)

    for plot_key in param_order:
        if plot_key not in plot_arrays:
            continue
        plot_info = plot_arrays[plot_key]
        print_summary_counts(plot_key, plot_info['cohort_animals'])
        print_summary_stats(plot_key, plot_info['pval_vs_zero'])
        figure = plot_summary_scatter(plot_info)
        utils.save_figure_multi_format(figure, summary_output_folder, get_plot_filename(plot_key), dpi=120)
        plt.close(figure)

    for plot_key, plot_info in plot_arrays.items():
        if plot_key in param_order:
            continue
        print_summary_counts(plot_key, plot_info['cohort_animals'])
        print_summary_stats(plot_key, plot_info['pval_vs_zero'])
        figure = plot_summary_scatter(plot_info)
        utils.save_figure_multi_format(figure, summary_output_folder, get_plot_filename(plot_key), dpi=120)
        plt.close(figure)


if __name__ == '__main__':
    main()
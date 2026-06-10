import os
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def get_gait_parameters_dir(path):
    if 'WT' in path:
        return os.path.join(path, 'gait parameters pixel_to_mm Jovin')
    return os.path.join(path, 'gait parameters')


def extract_animal_id(folder_name):
    session_match = re.match(r'^(.*?)\s+session\b', folder_name, flags=re.IGNORECASE)
    if session_match:
        return session_match.group(1).strip()
    print(f'Folder without session suffix, using full name as animal ID: {folder_name}')
    return folder_name.strip()


def get_animal_session_folders(gait_parameters_dir):
    folders_animals = []
    animal_session_folders = {}
    for folder_name in sorted(os.listdir(gait_parameters_dir)):
        folder_path = os.path.join(gait_parameters_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        folders_animals.append(folder_name)
        animal_id = extract_animal_id(folder_name)
        if animal_id not in animal_session_folders:
            animal_session_folders[animal_id] = []
        animal_session_folders[animal_id].append(folder_path)
    return folders_animals, animal_session_folders


def concatenate_session_rows(session_paths, filename):
    session_arrays = [np.load(os.path.join(session_path, filename)) for session_path in session_paths]
    return np.concatenate(session_arrays, axis=0)


def average_session_arrays(session_paths, filename):
    session_arrays = [np.load(os.path.join(session_path, filename)) for session_path in session_paths]
    return np.nanmean(np.stack(session_arrays, axis=0), axis=0)


def circular_mean_angles(angle_values, axis=0):
    sin_mean = np.nanmean(np.sin(angle_values), axis=axis)
    cos_mean = np.nanmean(np.cos(angle_values), axis=axis)
    mean_angles = np.arctan2(sin_mean, cos_mean)
    return np.mod(mean_angles, 2 * np.pi)


def average_session_phase_arrays(session_paths, filename):
    session_arrays = [np.load(os.path.join(session_path, filename)) for session_path in session_paths]
    return circular_mean_angles(np.stack(session_arrays, axis=0), axis=0)


def count_session_rows_by_speed(session_paths, filename, speed_values):
    session_array = concatenate_session_rows(session_paths, filename)
    session_df = pd.DataFrame({'speed': session_array[:, 1]})
    return session_df.groupby('speed').size().reindex(speed_values, fill_value=0).to_numpy()


def get_path_cohort(path, cohort_colors):
    for cohort_name in sorted(cohort_colors, key=len, reverse=True):
        if cohort_name in path:
            return cohort_name
    return None


def get_path_color(path, cohort_colors):
    cohort_name = get_path_cohort(path, cohort_colors)
    if cohort_name is not None:
        return cohort_colors[cohort_name]
    return 'gray'


def get_path_label(path, cohort_colors):
    cohort_name = get_path_cohort(path, cohort_colors)
    if cohort_name is not None:
        if cohort_name == 'WT':
            return 'non-inj'
        return cohort_name
    return 'unknown'


def build_support_dataframe(support_values, animal_ids, group_label, speed_values, support_column):
    support_rows = []
    for count_a, animal_id in enumerate(animal_ids):
        for speed_index, speed_value in enumerate(speed_values):
            support_value = support_values[count_a, speed_index, support_column]
            if np.isnan(support_value):
                continue
            support_rows.append(
                {
                    'animal_id': animal_id,
                    'group': group_label,
                    'speed': speed_value,
                    'support_value': support_value,
                }
            )
    return pd.DataFrame(support_rows)


def fit_support_mixedlm(support_df, prediction_speeds, support_title, group_label):
    clean_df = support_df.dropna(subset=['support_value', 'speed'])
    if clean_df.empty or clean_df['animal_id'].nunique() < 2 or clean_df['speed'].nunique() < 2:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            warnings.simplefilter('ignore', category=RuntimeWarning)
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            fit_result = smf.mixedlm(
                'support_value ~ speed',
                clean_df,
                groups=clean_df['animal_id'],
                re_formula='1',
            ).fit(reml=False, method='lbfgs', disp=False)
        intercept = fit_result.fe_params.get('Intercept', 0.0)
        slope = fit_result.fe_params.get('speed', 0.0)
        return intercept + slope * prediction_speeds
    except Exception as exc:
        print(f'Could not fit mixed model for {support_title} ({group_label}): {exc}')
        return None


def phase_to_percent(phase_values):
    return np.mod(phase_values, 2 * np.pi) * 100 / (2 * np.pi)


def build_phase_summary_dataframe(
    phase_values,
    animal_ids,
    group_label,
    pair_label,
    first_paw,
    second_paw,
    phase_paw_index,
    speed_values,
):
    first_values = phase_values[:, phase_paw_index[first_paw], :]
    second_values = phase_values[:, phase_paw_index[second_paw], :]
    pair_phase_angles = np.mod(second_values - first_values, 2 * np.pi)
    pair_phase_values = phase_to_percent(pair_phase_angles)
    phase_rows = []
    for count_a, animal_id in enumerate(animal_ids):
        for speed_index, speed_value in enumerate(speed_values):
            phase_value = pair_phase_values[count_a, speed_index]
            phase_angle = pair_phase_angles[count_a, speed_index]
            if np.isnan(phase_value):
                continue
            phase_rows.append(
                {
                    'animal_id': animal_id,
                    'group': group_label,
                    'pair_label': pair_label,
                    'speed': speed_value,
                    'phase_angle': phase_angle,
                    'phase_value': phase_value,
                }
            )
    return pd.DataFrame(phase_rows)


def build_phase_fit_dataframe(phase_df):
    if phase_df.empty:
        return phase_df

    grouped_rows = []
    for (animal_id, speed_value), group_df in phase_df.groupby(['animal_id', 'speed'], sort=True):
        mean_angle = circular_mean_angles(group_df['phase_angle'].to_numpy(), axis=0)
        grouped_rows.append(
            {
                'animal_id': animal_id,
                'speed': speed_value,
                'phase_angle': mean_angle,
            }
        )

    fit_df = pd.DataFrame(grouped_rows)
    if fit_df.empty:
        return fit_df

    reference_angle = circular_mean_angles(fit_df['phase_angle'].to_numpy(), axis=0)
    fit_df['fit_phase_value'] = phase_to_percent(reference_angle + np.angle(np.exp(1j * (fit_df['phase_angle'] - reference_angle))))
    fit_df['phase_value'] = phase_to_percent(fit_df['phase_angle'])
    return fit_df


def fit_phase_mixedlm(phase_df, prediction_speeds, phase_title, group_label, pair_label):
    response_column = 'fit_phase_value' if 'fit_phase_value' in phase_df.columns else 'phase_value'
    clean_df = phase_df.dropna(subset=[response_column, 'speed'])
    if clean_df.empty or clean_df['animal_id'].nunique() < 2 or clean_df['speed'].nunique() < 2:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            warnings.simplefilter('ignore', category=RuntimeWarning)
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            fit_result = smf.mixedlm(
                f'{response_column} ~ speed',
                clean_df,
                groups=clean_df['animal_id'],
                re_formula='1',
            ).fit(reml=False, method='lbfgs', disp=False)
        intercept = fit_result.fe_params.get('Intercept', 0.0)
        slope = fit_result.fe_params.get('speed', 0.0)
        return intercept + slope * prediction_speeds
    except Exception as exc:
        print(f'Could not fit mixed model for {phase_title} ({group_label}, {pair_label}): {exc}')
        return None


def configure_phase_polar_axis(ax, speed_values, title=None):
    if title:
        ax.set_title(title, fontsize=20, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_thetagrids([0, 90, 180, 270], labels=['0%', '25%', '50%', '75%'])
    ax.set_yticks(speed_values[::2])
    ax.set_yticklabels([
        f'{speed_value:.2f}'
        for speed_value in speed_values[::2]
    ])
    ax.set_rlabel_position(315)
    ax.set_ylim(speed_values[0] - 0.03, speed_values[-1] + 0.03)
    ax.grid(True, alpha=0.7)
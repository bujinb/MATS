#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Emissions → Exposure → Mortality pipeline (index-based merge)

Outputs:
- emissions_prepared.csv
- exposure_concentration.parquet   (isrm, Conc)
- exposure_grid.gpkg (layer='grid')  GeoDataFrame with geometry + Conc + census fields + Deaths
- exposure_grid.csv                 (flat CSV)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
import geopandas as gpd
from shapely.geometry import Point
from netCDF4 import Dataset
import numpy.ma as ma

# =========================
# CONFIG — EDIT PATHS HERE
# =========================
FP_EXEMPTED     = Path('/Users/bujin/Downloads/Average and Lowest fPM rates for exempted source (adapted from EPA-HQ-OAR-2018-0794-6919_attachment_1).xlsx')
FP_EGRID_PM     = Path('/Users/bujin/Downloads/egrid-draft-pm-emissions.xlsx')
FP_APPENDIXAB   = Path('/Users/bujin/Downloads/Appendix A and B from technical document.xlsx')
FP_PLANT_RATIOS = Path('/Users/bujin/Documents/EJ paper/Plantswithnoemisrate.csv')
FP_EGRID2023    = Path('/Users/bujin/Downloads/egrid2023_data_rev1.xlsx')

ISRM_SHP        = '/Users/bujin/Documents/CEE490/CEE 490 Data/ISRM.shp'
CENSUS_CSV      = '/Users/bujin/Documents/Old ISRM/Census/2020Pop-15+.csv'
MORTALITY_SHP   = '/Users/bujin/Documents/Inmap/Data/MortalityPop.shp'

# NetCDF kernels by stack height (meters): (path, low_bound_inclusive, high_bound_exclusive)
PM25_NETCDF = {
    'low' : ('/Users/bujin/Documents/Old ISRM/PM25L0.nc', 0, 57),
    'mid' : ('/Users/bujin/Documents/Old ISRM/PM25L1.nc', 57, 379),
    'high': ('/Users/bujin/Documents/Old ISRM/PM25L2.nc', 379, np.inf),
}
NC_VAR = 'PrimaryPM25'
SCALING_FACTOR = 28747.5637   # provided factor

# =========================
# HELPER SETTINGS
# =========================
pd.set_option('future.no_silent_downcasting', True)

# =========================
# HELPERS
# =========================
def to_stripped_str(s):
    return s.astype(str).str.strip()

def median_ignore_zeros(df, cols):
    return df[cols].replace(0, np.nan).median(axis=1, skipna=True)

def safe_mean(series):
    return pd.to_numeric(series, errors='coerce').mean(skipna=True)

def require_cols(df, cols, where):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {where}: {missing}")

def _strip_suffix(name):
    return name[:-2] if name.endswith(('_x','_y')) else name

def resolve_column(df, candidates, label):
    # exact first
    for c in candidates:
        if c in df.columns:
            return c
    # allow for _x/_y suffix
    stripped = {_strip_suffix(c): c for c in df.columns}
    for c in candidates:
        if c in stripped:
            return stripped[c]
    sample = ", ".join(map(str, df.columns[:30]))
    raise KeyError(f"Could not find column for {label}. Tried {candidates}. Columns include: {sample}")

# =========================
# 1) EXEMPTED UNITS + RATIO
# =========================
def load_exempted_and_ratio():
    exempt_cols = [
        'Plant state abbreviation', 'Plant name',
        'DOE/EIA ORIS plant or facility code', 'Unit ID', 'UniqueID_Final',
        'Latitude', 'Longitude', 'Average of All Data (lb/MMBtu)',
        'Average Annual Heat Input (mmBtu)'
    ]
    app = pd.read_excel(FP_EXEMPTED, sheet_name='Exempted', dtype=str)
    app = app[exempt_cols].copy()
    for c in ['Latitude', 'Longitude', 'Average of All Data (lb/MMBtu)', 'Average Annual Heat Input (mmBtu)']:
        app[c] = pd.to_numeric(app[c], errors='coerce')

    rate = pd.read_excel(FP_EXEMPTED, sheet_name='0.006 Limit Assumptions')
    require_cols(rate, ['UniqueID_Final', 'fPM Cost Effectiveness ($/ton)', 'fPM2.5 Cost Effectiveness ($/ton)'], '0.006 Limit Assumptions')
    rate = rate[['UniqueID_Final', 'fPM Cost Effectiveness ($/ton)', 'fPM2.5 Cost Effectiveness ($/ton)']].copy()
    rate['fPM_ratio'] = (
        pd.to_numeric(rate['fPM Cost Effectiveness ($/ton)'], errors='coerce') /
        pd.to_numeric(rate['fPM2.5 Cost Effectiveness ($/ton)'], errors='coerce')
    )
    rate = rate[['UniqueID_Final', 'fPM_ratio']]
    rate.loc[rate['fPM_ratio'] > 1, 'fPM_ratio'] = np.nan

    emission = app.merge(rate, on='UniqueID_Final', how='left')
    emission['fPM_ratio'] = emission['fPM_ratio'].fillna(emission['fPM_ratio'].mean(skipna=True))

    # Fill within plant
    cols_to_fill_by_plant = [
        'Latitude', 'Longitude',
        'Average of All Data (lb/MMBtu)',
        'Average Annual Heat Input (mmBtu)'
    ]
    emission[cols_to_fill_by_plant] = (
        emission.groupby('Plant name')[cols_to_fill_by_plant]
                .transform(lambda s: s.fillna(s.mean()))
    )

    nans_df = emission[emission['Latitude'].isna()].copy()
    emission = emission[~emission['Latitude'].isna()].copy()

    for k in ['DOE/EIA ORIS plant or facility code', 'Unit ID']:
        emission[k] = to_stripped_str(emission[k])
        nans_df[k]  = to_stripped_str(nans_df[k])

    return emission, nans_df

# =========================
# 2) EGRID PM (2018–2021)
# =========================
def load_egrid_pm_medians():
    years = [2018, 2019, 2020, 2021]
    sheet_names = {
        2018: '2018 PM Unit-level Data',
        2019: '2019 PM Unit-level Data',
        2020: '2020 PM Unit-level Data',
        2021: '2021 PM Unit-level Data'
    }
    yearly_dfs = []
    for y in years:
        df = pd.read_excel(FP_EGRID_PM, sheet_name=sheet_names[y])
        df.columns = [c.strip() for c in df.columns]
        need = [
            'Unit ID', 'DOE/EIA ORIS plant or facility code',
            'Unit unadjusted annual heat input (MMBtu)',
            'Unit unadjusted annual PM2.5 emissions (tons)',
            'Unit annual PM2.5 emission rate (lb/MMBtu)'
        ]
        require_cols(df, need, f'egrid {y}')
        df = df[need].copy()
        df.rename(columns={
            'Unit unadjusted annual heat input (MMBtu)': f'Annual Input {y}',
            'Unit unadjusted annual PM2.5 emissions (tons)': f'Emission {y}',
            'Unit annual PM2.5 emission rate (lb/MMBtu)': f'Emission Rate {y}'
        }, inplace=True)
        for k in ['DOE/EIA ORIS plant or facility code', 'Unit ID']:
            df[k] = to_stripped_str(df[k])
        for c in [f'Annual Input {y}', f'Emission {y}', f'Emission Rate {y}']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        yearly_dfs.append(df)

    merged_pm = reduce(
        lambda L, R: pd.merge(L, R, on=['DOE/EIA ORIS plant or facility code', 'Unit ID'], how='outer'),
        yearly_dfs
    )
    input_cols = [f'Annual Input {y}' for y in years]
    emis_cols  = [f'Emission {y}'      for y in years]
    rate_cols  = [f'Emission Rate {y}' for y in years]

    merged_pm['Avg Annual Input']  = median_ignore_zeros(merged_pm, input_cols)
    merged_pm['Avg Emissions']     = median_ignore_zeros(merged_pm, emis_cols)
    merged_pm['Avg Emission Rate'] = median_ignore_zeros(merged_pm, rate_cols)
    return merged_pm

# =========================
# 3) PATCH MISSING + STACKS
# =========================
def enrich_missing_with_appendix_ratios_stacks(nans_df, merged_pm):
    # Appendix A
    appendix_a = pd.read_excel(FP_APPENDIXAB, sheet_name='Appendix A')
    appendix_a.columns = appendix_a.columns.str.strip()
    uid_col = 'Unique ID'
    if uid_col not in appendix_a.columns:
        cands = [c for c in appendix_a.columns if c.strip().lower().startswith('unique id')]
        if not cands:
            raise KeyError("Could not find 'Unique ID' column in Appendix A.")
        uid_col = cands[0]
    require_cols(appendix_a, [uid_col, 'Min 99th'], 'Appendix A')
    appendix_a = appendix_a[[uid_col, 'Min 99th']].copy()
    uid = appendix_a[uid_col].astype(str).str.strip()
    appendix_a['DOE/EIA ORIS plant or facility code'] = uid.str.extract(r'^([^_]+)', expand=False).str.strip()
    appendix_a['Unit ID'] = uid.str.extract(r'_([^_]+)$', expand=False).str.strip()
    appendix_a.rename(columns={'Min 99th': 'Min_99th_EmissionRate'}, inplace=True)

    nans_df = nans_df.merge(
        appendix_a[['DOE/EIA ORIS plant or facility code', 'Unit ID', 'Min_99th_EmissionRate']],
        on=['DOE/EIA ORIS plant or facility code', 'Unit ID'], how='left'
    )

    # Fill emission rate from Appendix A when missing
    mask_missing = nans_df['Average of All Data (lb/MMBtu)'].isna()
    nans_df.loc[mask_missing, 'Average of All Data (lb/MMBtu)'] = nans_df.loc[mask_missing, 'Min_99th_EmissionRate']

    # Plant-level ratios
    ratios = pd.read_csv(FP_PLANT_RATIOS)
    require_cols(ratios, ['Plant name'], 'Plant ratios CSV')
    keep_cols = ['Plant name'] + [c for c in ['Ratio', 'Ratio (2017)'] if c in ratios.columns]
    ratios = ratios[keep_cols].drop_duplicates()
    nans_df = nans_df.merge(ratios, on='Plant name', how='left')

    # eGRID2023 stacks + plant latlon
    stack = pd.read_excel(FP_EGRID2023, sheet_name='UNT23')
    stack = stack[['DOE/EIA ORIS plant or facility code', 'Unit ID', 'Stack height (feet)']].copy()
    for k in ['DOE/EIA ORIS plant or facility code', 'Unit ID']:
        stack[k] = to_stripped_str(stack[k])
    stack['Stack height (feet)'] = pd.to_numeric(stack['Stack height (feet)'], errors='coerce')

    plant_latlon = pd.read_excel(FP_EGRID2023, sheet_name='PLNT23')
    plant_latlon = plant_latlon[['Plant name', 'Plant latitude', 'Plant longitude']].copy()
    nans_df = nans_df.merge(plant_latlon, on='Plant name', how='left')
    nans_df['Latitude']  = pd.to_numeric(nans_df['Latitude'], errors='coerce').fillna(pd.to_numeric(nans_df['Plant latitude'], errors='coerce'))
    nans_df['Longitude'] = pd.to_numeric(nans_df['Longitude'], errors='coerce').fillna(pd.to_numeric(nans_df['Plant longitude'], errors='coerce'))

    # Merge with PM medians
    for k in ['DOE/EIA ORIS plant or facility code', 'Unit ID']:
        nans_df[k]  = to_stripped_str(nans_df[k])
        merged_pm[k] = to_stripped_str(merged_pm[k])

    two = merged_pm.merge(nans_df, on=['DOE/EIA ORIS plant or facility code', 'Unit ID'], how='inner')

    # Impute missing emission rate: 2021 * Ratio(2017), else Avg Emission Rate
    mask_missing_rate = two['Average of All Data (lb/MMBtu)'].isna()
    rate_2021 = two.get('Emission Rate 2021', pd.Series(np.nan, index=two.index))
    ratio_2017 = two.get('Ratio (2017)', pd.Series(np.nan, index=two.index))
    two.loc[mask_missing_rate, 'Average of All Data (lb/MMBtu)'] = rate_2021[mask_missing_rate] * ratio_2017[mask_missing_rate]

    mask_still_missing = two['Average of All Data (lb/MMBtu)'].isna()
    two.loc[mask_still_missing, 'Average of All Data (lb/MMBtu)'] = two.loc[mask_still_missing, 'Avg Emission Rate']
    two['Average Annual Heat Input (mmBtu)'] = two['Avg Annual Input']

    return two, stack

# =========================
# 4) FINAL EMISSIONS TABLE
# =========================
def build_emissions_all(emission, two, stack):
    cols_final = [
        'Plant state abbreviation', 'Plant name',
        'DOE/EIA ORIS plant or facility code', 'Unit ID', 'UniqueID_Final',
        'Latitude', 'Longitude', 'Average of All Data (lb/MMBtu)',
        'Average Annual Heat Input (mmBtu)', 'fPM_ratio'
    ]
    two_subset = two.reindex(columns=cols_final, fill_value=np.nan)
    emissions_all = pd.concat([emission[cols_final], two_subset], ignore_index=True)
    emissions_all.drop_duplicates(subset=['DOE/EIA ORIS plant or facility code', 'Unit ID'], inplace=True, keep='first')
    emissions_all = emissions_all.merge(stack, on=['DOE/EIA ORIS plant or facility code', 'Unit ID'], how='left')

    # Fill stack height and cast numerics (avoids warnings)
    emissions_all['Stack height (feet)'] = pd.to_numeric(emissions_all['Stack height (feet)'], errors='coerce')
    mean_stack = safe_mean(emissions_all['Stack height (feet)'])
    emissions_all['Stack height (feet)'] = emissions_all['Stack height (feet)'].fillna(mean_stack)

    for c in ['Average of All Data (lb/MMBtu)', 'Average Annual Heat Input (mmBtu)', 'Latitude', 'Longitude', 'fPM_ratio']:
        emissions_all[c] = pd.to_numeric(emissions_all[c], errors='coerce')
    for k in ['DOE/EIA ORIS plant or facility code', 'Unit ID', 'Plant name', 'Plant state abbreviation']:
        if k in emissions_all.columns:
            emissions_all[k] = to_stripped_str(emissions_all[k])

    return emissions_all

# =========================
# 5) EMISSIONS DELTAS (TONS)
# =========================
def compute_emission_deltas(emis):
    emis = emis.copy()
    emis['Adjusted Emission Rate (lb/MMBtu)'] = np.clip(emis['Average of All Data (lb/MMBtu)'], a_min=None, a_max=0.01)
    # Tons = (lb/MMBtu * MMBtu)/2000; factor 2 * fPM_ratio retained
    emis['Emissions Would'] = (emis['Adjusted Emission Rate (lb/MMBtu)'] * emis['Average Annual Heat Input (mmBtu)'] / 2000.0) * 2.0 * emis['fPM_ratio']
    emis['Emissions Now']   = (emis['Average of All Data (lb/MMBtu)']     * emis['Average Annual Heat Input (mmBtu)'] / 2000.0) * 2.0 * emis['fPM_ratio']
    emis['Emissions Tons Change'] = emis['Emissions Now'] - emis['Emissions Would']
    return emis

# =========================
# 6) MAP TO ISRM + KERNEL
# =========================
def to_isrm_concentration(emis):
    # Build point geometry
    emis = emis.copy()
    emis['geometry'] = [Point(xy) for xy in zip(emis['Longitude'], emis['Latitude'])]
    emis_gdf = gpd.GeoDataFrame(emis, geometry='geometry', crs='EPSG:4326')

    # ISRM grid
    isrm_gdf = gpd.read_file(ISRM_SHP)
    if isrm_gdf.crs is None:
        raise ValueError("ISRM shapefile lacks CRS.")
    isrm_gdf = isrm_gdf.to_crs(emis_gdf.crs)

    # Normalize id column to 'isrm'
    id_col = None
    for cand in ['isrm', 'cell_id', 'index', 'grid_id', 'ID']:
        if cand in isrm_gdf.columns:
            id_col = cand
            break
    if id_col is None:
        raise ValueError("No ISRM id column found in ISRM shapefile.")
    if id_col != 'isrm':
        isrm_gdf = isrm_gdf.rename(columns={id_col: 'isrm'})

    # Spatial join points → ISRM cells
    gridcell = gpd.sjoin(emis_gdf, isrm_gdf[['isrm', 'geometry']], how='inner', predicate='intersects')

    # Stack height meters
    gridcell['Stack height (m)'] = gridcell['Stack height (feet)'] * 0.3048

    def cond(h, low, high):
        return (h >= low) & (h < high)

    total_result = None
    for label, (nc_file, low, high) in PM25_NETCDF.items():
        subset = gridcell[cond(gridcell['Stack height (m)'], low, high)]
        if subset.empty:
            continue

        grp = subset.groupby('isrm', as_index=False)['Emissions Tons Change'].sum()

        with Dataset(nc_file) as ds:
            kernel = ma.getdata(ds.variables[NC_VAR][0])

        # assumes grp['isrm'] are valid integer indices for 'kernel'
        results = []
        for i, e_val in zip(grp['isrm'].astype(int).values, grp['Emissions Tons Change'].values):
            results.append(kernel[i] * float(e_val))

        bin_sum = np.sum(results, axis=0) * SCALING_FACTOR
        total_result = bin_sum if total_result is None else (total_result + bin_sum)

    if total_result is None:
        raise RuntimeError("No units fell into any stack-height bin.")

    conc = pd.DataFrame({'Conc': total_result}).reset_index().rename(columns={'index': 'isrm'})

    # Sanity check: isrm ids within kernel bounds?
    N = len(isrm_gdf)
    bad_ids = conc.loc[(conc['isrm'] < 0) | (conc['isrm'] >= N), 'isrm'].unique()
    if len(bad_ids):
        raise RuntimeError(f"Found {len(bad_ids)} concentration cells with out-of-range ISRM ids (min=0, max={N-1}).")

    return conc, isrm_gdf

# =========================
# 7) JOIN MORTALITY + CENSUS ON INDEX, COMPUTE DEATHS
# =========================
def join_by_index_and_compute_deaths(conc, ca=None):
    """
    Implements exactly your requested sequence:
    - ensure conc has 'index' key,
    - mortality = read shp, reset_index(), drop MortalityR/TotalPop,
    - census: rename isrm->index,
    - merge on 'index',
    - compute Deaths with Orellano HR=1.095 per 10 µg/m³.
    """
    # ensure conc has 'index' key
    if 'index' in conc.columns:
        conc_idx = conc.copy()
    elif 'isrm' in conc.columns:
        conc_idx = conc.rename(columns={'isrm': 'index'})
    else:
        conc_idx = conc.reset_index().rename(columns={'index': 'index'})

    # read mortality, make 'index', drop MortalityR/TotalPop (will come from census)
    mortality = gpd.read_file(MORTALITY_SHP).reset_index(drop=True)
    mortality = mortality.reset_index()  # adds 'index'
    mortality = mortality.drop(columns=['MortalityR', 'TotalPop'], errors='ignore')

    # read census, rename key to 'index' if needed
    census = pd.read_csv(CENSUS_CSV).reset_index(drop=True)
    if 'index' not in census.columns and 'isrm' in census.columns:
        census = census.rename(columns={'isrm': 'index'})

    # merge
    mortality = mortality.merge(census, on='index', how='left').fillna(0)
    df = conc_idx.merge(mortality, on='index', how='left')

    # CRS selection
    cacrs = ca.crs if ca is not None else mortality.crs
    df = gpd.GeoDataFrame(df, geometry='geometry', crs=cacrs)

    # robustly resolve population and baseline rate columns
    pop_col  = resolve_column(df, ['TotalPop','Total_Pop','Population','Pop15plus','TotalPop_15plus'], 'population (TotalPop)')
    mort_col = resolve_column(df, ['MortalityR','MortRate','MortalityRate','baseline_rate','y0'], 'baseline mortality rate (MortalityR)')

    # ensure numeric
    df['Conc']   = pd.to_numeric(df['Conc'], errors='coerce').fillna(0.0)
    df[pop_col]  = pd.to_numeric(df[pop_col], errors='coerce').fillna(0.0)
    df[mort_col] = pd.to_numeric(df[mort_col], errors='coerce').fillna(0.0)

    # compute deaths (Orellano HR 1.095 per 10 µg/m³)
    df['Deaths'] = (
        (np.exp(np.log(1.095) / 10.0 * df['Conc']) - 1.0)
        * df[pop_col]
        * df[mort_col]
        / 100000.0
    )

    return df

# =========================
# MAIN
# =========================
def main():
    # Prep emissions
    emission, nans_df = load_exempted_and_ratio()
    merged_pm = load_egrid_pm_medians()
    two, stack = enrich_missing_with_appendix_ratios_stacks(nans_df, merged_pm)
    emissions_all = build_emissions_all(emission, two, stack)
    print(f"[INFO] Final emissions rows: {len(emissions_all):,}")
    emissions_all.to_csv('emissions_prepared.csv', index=False)

    # Emissions deltas
    emis_deltas = compute_emission_deltas(emissions_all)

    # To concentration via ISRM
    conc, isrm_gdf = to_isrm_concentration(emis_deltas)
    print(f"[INFO] Concentration array length: {len(conc):,}")
    conc.to_parquet('exposure_concentration.parquet', index=False)

    # Build exposure grid (your index-based path) and compute deaths
    # If you have a 'ca' GeoDataFrame with a desired CRS, pass it; else leave None.
    exposure_grid = join_by_index_and_compute_deaths(conc, ca=None)
    print(f"[INFO] Exposure grid rows: {len(exposure_grid):,}")

    # Save outputs
    exposure_grid.to_file('exposure_grid.gpkg', layer='grid', driver='GPKG')
    exposure_grid.to_csv('exposure_grid.csv', index=False)

    print("[DONE] Saved: emissions_prepared.csv, exposure_concentration.parquet, exposure_grid.gpkg, exposure_grid.csv")

if __name__ == '__main__':
    main()

# here some feature name are defined as constant values

key = ['city', 'year', 'weekofyear', 'week_start_date']

target = ["total_cases"]
log1p_target = ["log1p_total_cases"]
clip_target = ["clip_total_cases"]
ndvi = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']
reanalysis_temp = [
    'reanalysis_air_temp_k',
    'reanalysis_avg_temp_k',
    'reanalysis_dew_point_temp_k',
    'reanalysis_max_air_temp_k',
    'reanalysis_min_air_temp_k'
]
reanalysis_tdtr_k = ['reanalysis_tdtr_k']
reanalysis_humidity = [
    'reanalysis_relative_humidity_percent',
    'reanalysis_specific_humidity_g_per_kg'
]
reanalysis_precip = [
  'reanalysis_precip_amt_kg_per_m2',
  'reanalysis_sat_precip_amt_mm',
]

reanalysis = reanalysis_temp + reanalysis_humidity + reanalysis_precip + reanalysis_tdtr_k

station_temp = [
    'station_avg_temp_c',
    'station_diur_temp_rng_c',
    'station_max_temp_c',
    'station_min_temp_c'
]

station = station_temp + ['station_precip_mm']

precipitation = ['precipitation_amt_mm']

orig_feats = ndvi + precipitation + reanalysis + station

def get_roll_feats(cols, n):
    return ["roll{0}_".format(n) + col for col in cols]

def get_lag_feats(cols, nlist):
    return [col + "_lag{0}".format(i) for col in cols for i in nlist]

def get_scaled_feats(cols):
    return ["scale_{0}".format(col) for col in cols]

def get_pca_feats(n):
    return ["pca_{0}".format(col) for col in range(1, n+1)]

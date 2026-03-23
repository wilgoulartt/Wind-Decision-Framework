# run_sigahp.py
import math
import ee
import geemap
import geopandas as gpd
from geobr import read_municipality

# =========================================================
# 0) CONFIG 
# =========================================================
PROJECT_ID = "SEU_PROJETO_GEE"  
SCALE = 500
START_DATE = "2000-01-01"
END_DATE   = "2024-12-31"

# Exclusões
SLOPE_EXCLUDE_DEG = 15
VEL_MIN_WS80      = 6.0
BUF_URB_M  = 500
BUF_AGUA_M = 200
BUF_UC_M   = 300
BUF_AGRI_M = 500

# Pesos AHP (somam 1.0)
W_VENTO, W_SLOPE, W_URB, W_UC, W_AGRI = 0.41, 0.18, 0.17, 0.14, 0.10

# MapBiomas
URBANA = 24
AGUA   = 33
AGRI_CODES = [49]  # ajustar conforme sua legenda

# Correção GWA
WSPD_GWA_MEAN = 8.3

# =========================================================
# 1) INIT GEE
# =========================================================
ee.Initialize(project=PROJECT_ID)

# =========================================================
# 2) ROI — RNF
# =========================================================
def get_roi_rnf():
    municipios_rnf = [
        "Campos dos Goytacazes","Macaé","Carapebus","Cardoso Moreira",
        "Conceição de Macabu","Quissamã","São Fidélis",
        "São Francisco de Itabapoana","São João da Barra"
    ]
    rj = read_municipality(year=2023, code_muni="RJ")
    rnf_gdf = rj[rj["name_muni"].isin(municipios_rnf)].to_crs("EPSG:4326").dissolve()
    rnf_fc = geemap.geopandas_to_ee(rnf_gdf)
    roi = ee.FeatureCollection(rnf_fc).geometry()
    return roi

roi = get_roi_rnf()

# =========================================================
# 3) HELPERS (robustos)
# =========================================================
def _first_or_zero_from_values(dct):
    vals = ee.List(ee.Dictionary(dct).values())
    return ee.Number(ee.Algorithms.If(vals.size().gt(0), vals.get(0), 0))

def minmax01_named(img, band_name):
    stats = ee.Dictionary(
        img.reduceRegion(ee.Reducer.minMax(), roi, SCALE, bestEffort=True, maxPixels=1e9)
    )
    key_min = band_name + "_min"
    key_max = band_name + "_max"
    mn = ee.Number(ee.Algorithms.If(stats.contains(key_min), stats.get(key_min), 0))
    mx = ee.Number(ee.Algorithms.If(stats.contains(key_max), stats.get(key_max), 1))
    denom = mx.subtract(mn)
    norm = ee.Image(ee.Algorithms.If(denom.neq(0), img.subtract(mn).divide(denom), ee.Image(0)))
    return norm.clamp(0, 1)

def buffer_from_mask(img01, radius_m):
    img01 = img01.unmask(0).gt(0)
    count_dict = img01.reduceRegion(ee.Reducer.sum(), roi, SCALE, bestEffort=True, maxPixels=1e9)
    cnt = _first_or_zero_from_values(count_dict)
    return ee.Image(ee.Algorithms.If(
        cnt.gt(0),
        img01.distance(ee.Kernel.euclidean(radius_m, "meters")).lte(radius_m),
        ee.Image(0)
    ))

def safe_distance(img01, max_radius_m, out_name):
    img01 = img01.unmask(0).gt(0)
    count_dict = img01.reduceRegion(ee.Reducer.sum(), roi, SCALE, bestEffort=True, maxPixels=1e9)
    cnt = _first_or_zero_from_values(count_dict)
    dist = img01.distance(ee.Kernel.euclidean(max_radius_m, "meters")).rename(out_name).clip(roi)
    return ee.Image(ee.Algorithms.If(cnt.gt(0), dist, ee.Image(max_radius_m).rename(out_name).clip(roi)))

# =========================================================
# 4) CAMADAS BASE
# =========================================================
# SRTM + slope
srtm  = ee.Image("USGS/SRTMGL1_003").clip(roi)
slope = ee.Terrain.slope(srtm).rename("slope").clip(roi)

# MapBiomas 2023
mb = (ee.Image("projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1")
      .select("classification_2023").clip(roi))

mask_urb  = mb.eq(URBANA).unmask(0)
mask_agua = mb.eq(AGUA).unmask(0)

mask_agri = ee.Image(0)
for code in AGRI_CODES:
    mask_agri = mask_agri.Or(mb.eq(code))
mask_agri = mask_agri.unmask(0).rename("mask_agri")

# UCs (WDPA não-marinho)
uc_fc = (ee.FeatureCollection("WCMC/WDPA/current/polygons")
         .filterBounds(roi)
         .filter(ee.Filter.eq("MARINE", 0)))
uc_count = uc_fc.size()

uc_mask = ee.Image.constant(0).rename("uc_mask")
uc_mask = ee.Image(ee.Algorithms.If(
    uc_count.gt(0),
    ee.Image.constant(0).byte().paint(uc_fc, 1).unmask(0),
    ee.Image(0).rename("uc_mask")
))

# =========================================================
# 5) VENTO ERA5 (10m -> 80m) + correção GWA
# =========================================================
era5 = (ee.ImageCollection("ECMWF/ERA5/MONTHLY")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(roi))

def ws10(i):
    u = i.select("u_component_of_wind_10m")
    v = i.select("v_component_of_wind_10m")
    return u.hypot(v).rename("ws10")

ws10_mean = era5.map(ws10).mean().clip(roi)

# Perfil log (80m a partir de 10m)
z, zr, z0 = 80.0, 10.0, 0.1
factor_log = math.log(z / z0) / math.log(zr / z0)
ws80 = ws10_mean.multiply(factor_log).rename("ws80")

# fator de correção regional (constante)
ws80_mean = ee.Number(ws80.reduceRegion(
    ee.Reducer.mean(), roi, SCALE, bestEffort=True, maxPixels=1e9
).get("ws80"))

F_CORR = ee.Number(WSPD_GWA_MEAN).divide(ws80_mean)
ws80_corr = ws80.multiply(F_CORR).rename("ws80_corr").clip(roi)

# =========================================================
# 6) EXCLUSÃO (E)
# =========================================================
urb_excl  = buffer_from_mask(mask_urb,  BUF_URB_M).rename("ex")
agua_excl = buffer_from_mask(mask_agua, BUF_AGUA_M).rename("ex")
agri_excl = buffer_from_mask(mask_agri, BUF_AGRI_M).rename("ex")

uc_excl = ee.Image(ee.Algorithms.If(
    uc_count.gt(0),
    buffer_from_mask(uc_mask, BUF_UC_M).rename("ex"),
    ee.Image(0).rename("ex")
))

slope_excl = slope.gt(SLOPE_EXCLUDE_DEG).rename("ex")
wind_excl  = ws80_corr.lt(VEL_MIN_WS80).rename("ex")

# exclusão total
exclusion_full = (urb_excl.max(agua_excl).max(agri_excl).max(uc_excl).max(slope_excl).max(wind_excl)
                  .clip(roi).selfMask().rename("exclusion_full"))

# E = 1 viável | 0 excluído
E = ee.Image(1).where(exclusion_full.eq(1), 0).rename("E").clip(roi).selfMask()

# =========================================================
# 7) CRITÉRIOS (0–1) + S01 + A01
# =========================================================
dist_urban = safe_distance(mask_urb,  20000, "d_urb")
dist_uc    = safe_distance(uc_mask,   50000, "d_uc")
dist_agri  = safe_distance(mask_agri, 30000, "d_agri")

wind01  = minmax01_named(ws80_corr, "ws80_corr")
slope01 = ee.Image(1).subtract(minmax01_named(slope, "slope"))
nurb01  = minmax01_named(dist_urban, "d_urb")
nuc01   = minmax01_named(dist_uc, "d_uc")
nagri01 = minmax01_named(dist_agri, "d_agri")

S01 = (wind01.multiply(W_VENTO)
       .add(slope01.multiply(W_SLOPE))
       .add(nurb01.multiply(W_URB))
       .add(nuc01.multiply(W_UC))
       .add(nagri01.multiply(W_AGRI))
      ).clamp(0, 1).rename("S01")

A01 = S01.multiply(E).rename("A01")

# Classes (limiares fixos — versão produto)
A_class = (ee.Image(0)
    .where(A01.gt(0).And(A01.lt(0.30)), 1)
    .where(A01.gte(0.30).And(A01.lt(0.50)), 2)
    .where(A01.gte(0.50).And(A01.lt(0.70)), 3)
    .where(A01.gte(0.70).And(A01.lt(0.80)), 4)
    .where(A01.gte(0.80), 5)
).updateMask(A01.gt(0)).rename("A_class")

# =========================================================
# 8) EXPORTS (escolha 1 método)
# =========================================================
def export_to_drive(img, desc, folder="GEE_Exports_RNF", prefix=None):
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=desc,
        folder=folder,
        fileNamePrefix=prefix or desc,
        region=roi,
        scale=SCALE,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task

# Exemplo:
# export_to_drive(S01.float(), "S_Adequabilidade_SIG_AHP", prefix="S_Adequabilidade_SIG_AHP")
# export_to_drive(A01.float(), "A_Area_Adequada", prefix="A_Area_Adequada")
# export_to_drive(A_class.byte(), "Classes_Aptidao_Final", prefix="Classes_Aptidao_Final")

# =========================================================
# 9) (Opcional) VISUALIZAÇÃO RÁPIDA (Map)
# =========================================================
# Map = geemap.Map()
# Map.add_basemap("SATELLITE")
# Map.addLayer(S01, {"min":0, "max":1}, "S01", True)
# Map.addLayer(A_class, {"min":1, "max":5}, "Classes", True)
# Map.centerObject(roi, 8)
# display(Map)

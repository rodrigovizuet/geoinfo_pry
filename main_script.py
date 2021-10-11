'''
####################################
## Accesibilidad a taller de bici ##
####################################
'''
#%% Paquetería
import pandas as pd
import geopandas as gpd
from shapely.geometry import *
from access import Access, weights
import matplotlib.pyplot as plt
import matplotlib
import contextily as ctx
import os

os.chdir('/home/lvizuet/Git/geoinfo_pry/')

# Funciones
def multi_line(line, dista):
    if line.geom_type == 'MultiLineString':
        segments = []
        for l in line:
            segments += break_line(l, dista)
    else:
        segments = break_line(line, dista)
    return segments

def break_line(line, dist):
    line = LineString([xy[0:2] for xy in list(line.coords)])
    if line.length <= dist:
        return [line]
    else: 
        segments = cut(line, dist)
        return [segments[0]] + break_line(segments[1], dist)

def cut(line, distance):
    if distance <= 0.0 or distance >= line.length:
        return [line]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        p = line.project(Point(p)) 
        if p == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if p > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

#%% Ciclovias
ciclo = gpd.read_file('data/ciclovias_cdmx.zip')
ciclo_lines = pd.DataFrame({'ID':[], 'NOMBRE':[], 'TIPO_IC':[], 
                            'VIALIDAD':[], 'TIPO_VIA':[], 'ESTADO':[], 
                            'SENTIDO':[], 'INSTANCIA':[], 'AÑO':[],'geometry':[]})
for i in range(ciclo.shape[0]):
    df_l = pd.DataFrame({'ID':ciclo.iloc[i,0], 'NOMBRE':ciclo.iloc[i,1],
                         'TIPO_IC':ciclo.iloc[i,2], 
                         'VIALIDAD':ciclo.iloc[i,3],
                         'TIPO_VIA':ciclo.iloc[i,4], 
                         'ESTADO':ciclo.iloc[i,5], 
                         'SENTIDO':ciclo.iloc[i,6],
                         'INSTANCIA':ciclo.iloc[i,7], 
                         'AÑO':ciclo.iloc[i,8],
                         'geometry':(multi_line(ciclo.iloc[i,-1], 1000))})
    ciclo_lines = ciclo_lines.append(df_l)
    
ciclo_lines = gpd.GeoDataFrame(ciclo_lines)
ciclo_lines.reset_index(drop=True, inplace=True)
ciclo_lines.reset_index(inplace=True)
ciclo_lines.columns = [x.lower() for x in ciclo_lines.columns]
ciclo_lines.rename(columns={'index':'fid', 'id':'fid_org', 'año':'year'}, 
                   inplace=True)
ciclo_centroids = ciclo_lines.copy()
ciclo_centroids.loc[:,'geometry'] = ciclo_centroids.geometry.centroid

#Generar variable con codigo unico para cada punto
ciclo_centroids = ciclo_centroids.rename(columns={'fid_org':'id_ciclovia', 'fid':'id_centroid'})

#Exportar gpd's
ciclo_lines.to_file('products/ciclovias_lines.gpkg', driver='GPKG', layer='ciclovias')
ciclo_centroids.to_file('products/ciclovias_centroids.gpkg', driver='GPKG', layer='ciclovias_ctr')
del ciclo, df_l, i



#%% Modelar el valor de demanda por densidad población alrededor (knn5)
cdmx = gpd.read_file("data/agebs_cdmx_2020.zip").to_crs(32614)[['CVEGEO', 'geometry']]
cdmx.iloc[:,1] = cdmx.geometry.centroid

pob = pd.read_csv("data/conjunto_de_datos_ageb_urbana_09_cpv2020.zip", dtype={'ENTIDAD':str, 'MUN':str, 'LOC':str, 'AGEB':str})
pob['CVEGEO'] = pob['ENTIDAD'] + pob['MUN'] + pob['LOC'] + pob['AGEB']
pob = pob[['CVEGEO','POBTOT']]
pob = gpd.GeoDataFrame(pob.merge(cdmx, on='CVEGEO'))

cl = gpd.GeoDataFrame(gpd.read_file('products/ciclovias_centroids.gpkg'), geometry='geometry', crs=32614)
cl_buff = cl.copy()[['id_centroid','geometry']]
cl_buff['geometry'] = cl_buff.geometry.buffer(3750) # A 15 minutos en bici (proemdio de vel 15km/h)

den = gpd.sjoin(pob, cl_buff, how='right', op='within')[['CVEGEO','POBTOT','id_centroid']]
den = den.groupby('id_centroid', as_index=False).POBTOT.sum()
den = den.merge(cl, on='id_centroid')

den = gpd.GeoDataFrame(den[['id_centroid','POBTOT','geometry']]).to_file('products/ciclovias_centroids_demand.gpkg', driver='GPKG', layer='ciclovias')

#%% Modelo de accesibilidad
cl = gpd.read_file('products/ciclovias_centroids_demand.gpkg')
ta = gpd.read_file('Talleres_Bici.gpkg')
ta['per_ocu'] = ta.per_ocu.apply(lambda x: 1 if x=='0 a 5 personas' else 2)
ta = ta.to_crs(32614)

cost = cl[['id_centroid']].merge(ta[['id']], how='cross')
cost['cost'] = 0
cost.columns = ['origen', 'destino', 'cost']
# Instanciamos un objeto de la clase Access
A = Access(demand_df            = cl,
           demand_index         = 'id_centroid',
           demand_value         = 'POBTOT',
           supply_df            = ta,
           supply_index         = 'id',
           supply_value         = 'per_ocu',
           cost_df              = cost,
           cost_origin          = 'origen',
           cost_dest            = 'destino',
           cost_name            = 'cost')

# Calculamos las distancias
A.create_euclidean_distance(threshold = 250000, centroid_o = True, centroid_d = True)
cost = A.cost_df[['origen','destino','euclidean']]
cost.to_csv('products/euclidian_cost.csv')

A = Access(demand_df            = cl,
           demand_index         = 'id_centroid',
           demand_value         = 'POBTOT',
           supply_df            = ta,
           supply_index         = 'id',
           supply_value         = 'per_ocu',
           cost_df              = cost,
           cost_origin          = 'origen',
           cost_dest            = 'destino',
           cost_name            = 'euclidean')

gaussian = weights.gaussian(750)
gravity = weights.gravity(scale = 750, alpha = -1)
A.weighted_catchment(name = "gravity",  weight_fn = gaussian)
A.raam(name = "raam", tau = 750)

# Guardamos los modelos por si los queremos reutilizar
A.norm_access_df.to_csv("products/accesibilidad_distancia_euclidiana.csv")

mapa_accesibilidad = cl.set_index('id_centroid')[['geometry']].join(A.norm_access_df, how = "inner")
mapa_accesibilidad.columns = ['geometry','gravity','raam']

fig, ax = plt.subplots(1,2, figsize=(20,15))
cmap = matplotlib.cm.viridis
mapa_accesibilidad.to_crs(epsg=3857).plot('raam', legend = True,
                                          cmap =  cmap.reversed(), 
                                          markersize = 7, alpha = 0.6, ax = ax[0],
                                          vmin = mapa_accesibilidad['raam'].quantile(0.05), 
                                          vmax = mapa_accesibilidad['raam'].quantile(0.95),
                                          )
mapa_accesibilidad.to_crs(epsg=3857).plot('gravity', legend = True,
                                          cmap =  cmap, 
                                          markersize = 7, alpha = 0.6, ax = ax[1],
                                          vmin = mapa_accesibilidad['gravity'].quantile(0.05), 
                                          vmax = mapa_accesibilidad['gravity'].quantile(0.95),
                                          )
ax[0].set(title='Modelo RAAM')
ax[1].set(title='Modelo Gravitatorio')
for i in range(len(ax)):
    ax[i].set_axis_off()
    ctx.add_basemap(ax[i], source=ctx.providers.CartoDB.Positron)    
plt.tight_layout()
plt.show()

mapa_accesibilidad.to_crs(32614).to_file('products/ciclovias_accesibilidad.gpkg', driver='GPKG', layer='accesibilidad')

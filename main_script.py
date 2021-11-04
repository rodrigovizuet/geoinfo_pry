#%% Modelo de accesibilidad 
'''
####################################
## Accesibilidad a taller de bici ##
####################################
'''
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.neighbors import BallTree
from shapely.geometry import *
from shapely.ops import nearest_points
from access import access, weights
import matplotlib.pyplot as plt
import matplotlib
import contextily as ctx
import warnings
warnings.filterwarnings("ignore")
import os
os.chdir('C:/lrgv/Git/geoinfo_pry/')

#%% Funciones accesibilidad
'''
FUNCIONES PARA MODELO DE ACCESIBILIDAD
'''
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

def KN(gdA, gdB, knn=5,dist_lim=1000):
    nA = list(gdA.geometry.apply(lambda x: (x.x, x.y)))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = BallTree(nB)
    dist, idx = btree.query(np.array(nA), k=knn)
    idx = [list(x) for x in list(idx)]
    for i in range(dist.shape[0]):
        rev_j = list(range(dist.shape[1]))
        rev_j.sort(reverse=True)
        for j in rev_j:
            if dist[i,j]>dist_lim:
                del idx[i][j]
                
    rev_i = list(range(dist.shape[0]))
    rev_i.sort(reverse=True)
    for i in rev_i:
        if len(idx[i])==0:
            del idx[i]
            del nA[i]
    return [Point(x) for x in nA], idx

#%% Ciclovias
ciclo = gpd.read_file('data/ciclovias_cdmx.zip').to_crs(32614)
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
    
ciclo_lines = gpd.GeoDataFrame(ciclo_lines, crs=32614)
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

#%% Modelo de demanda
cdmx = gpd.read_file("data/agebs_cdmx_2020.zip").to_crs(32614)[['CVEGEO', 'geometry']]
cdmx.iloc[:,1] = cdmx.geometry.centroid
cl = gpd.GeoDataFrame(gpd.read_file('products/ciclovias_centroids.gpkg'), geometry='geometry', crs=32614)[['id_centroid','geometry']]

pob = pd.read_csv("data/conjunto_de_datos_ageb_urbana_09_cpv2020.zip", dtype={'ENTIDAD':str, 'MUN':str, 'LOC':str, 'AGEB':str})
pob['CVEGEO'] = pob['ENTIDAD'] + pob['MUN'] + pob['LOC'] + pob['AGEB']
pob = pob[['CVEGEO','P_8A14','P_15A17','P_18YMAS','P_60YMAS']]
for i in ['P_8A14','P_15A17','P_18YMAS','P_60YMAS']:
    pob[i] = pob[i].replace('*',1)
    pob[i] = pob[i].replace('N/D',np.nan)
    pob[i] = pob[i].astype(float)

pob['POBTOT'] = pob['P_8A14'] + pob['P_15A17']+pob['P_18YMAS']-pob['P_60YMAS']
pob = pob[['CVEGEO', 'POBTOT']]
pob = gpd.GeoDataFrame(pob.merge(cdmx, on='CVEGEO'))

points, knn_points = KN(cl, pob, 3, 1500)

count = pd.DataFrame({'idx':[j for i in knn_points for j in i]}).value_counts().reset_index()
count.columns = ['idx','kn']
count.set_index('idx', inplace=True)

pob = pd.concat([pob,count], axis=1)
pob['kn'] = pob.kn.fillna(1)
pob['pob_pond'] = pob['POBTOT']/pob['kn']


# YA NO SE QUE MAS PONER.....  HAY CEROS EN ALGUNAS POBLACIONES

for i in range(len(points)):
    cl.loc[cl.geometry==points[i], 'POBTOT'] = int(pob.iloc[knn_points[i],-1].sum())

ax = cl[cl.POBTOT==0].plot()
cl.plot(ax=ax, markersize=.5)


den = gpd.sjoin(pob, cl_buff, how='right', op='within')[['CVEGEO','POBTOT','id_centroid']]
den = den.groupby('id_centroid', as_index=False).POBTOT.sum()
den = den.merge(cl, on='id_centroid')

#den = gpd.GeoDataFrame(den[['id_centroid','POBTOT','geometry']]).to_file('products/ciclovias_centroids_demand.gpkg', driver='GPKG', layer='ciclovias')

#%% Modelo de accesibilidad
cl = gpd.read_file('products/ciclovias_centroids_demand.gpkg')
ta = gpd.read_file('Talleres_Bici.gpkg')
ta['per_ocu'] = ta.per_ocu.apply(lambda x: 1 if x=='0 a 5 personas' else 2)
ta = ta.to_crs(32614)

cost = pd.merge(cl[['id_centroid']], ta[['id']], left_on='id_centroid',
                right_on='id', how='outer')
cost['cost'] = 0
cost.columns = ['origen', 'destino', 'cost']
# Instanciamos un objeto de la clase Access
A = access(demand_df            = cl,
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

A = access(demand_df            = cl,
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


#%% Busqueda
'''
########################
## Indices espaciales ##
########################
'''
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from time import process_time
import warnings
warnings.filterwarnings("ignore")
import os
os.chdir('C:/lrgv/Git/geoinfo_pry/')

#%% Funciones de QuadTree
'''
FUNCIONES PARA QUADTREES
'''
def espacial(puntos, poligono):
    if len(set(poligono.index.duplicated()))>1:
        print('CUIDADO, tu poligonos tienen índices repetidos')
    si = puntos.sindex
    
    pmi = []
    ide = []
    for poly in poligono.itertuples():
        c = list(si.intersection(poly.geometry.bounds))
        pmi += c
        ide += [poly.Index]*len(c)

    pm = puntos.iloc[list(set(pmi))]

    result = pm.loc[pm.intersects(poligono.unary_union)]
    result['id_busqueda'] = ide
    return result

def normal(puntos, poligono):    
    pm = []
    ide = []
    for poly in poligono.itertuples():
        c = list(puntos.loc[puntos.intersects(poly.geometry)].index)
        pm += c
        ide += [poly.Index]*len(c)

    result = puntos.loc[puntos.index.isin(list(set(pm)))]
    result['id_busqueda'] = ide
    return result

def buscador(tramo_ciclovia, talleres, zonas, usar_spindex=True):
    if usar_spindex:
        step1 = espacial(tramo_ciclovia, zonas).id_busqueda.unique()
        if len(step1)>0:
            return espacial(talleres, zonas.iloc[step1, :])
        else:
            print('Lo siento, este tramo de ciclovía no está en la CDMX')
    else:
        step1 = normal(tramo_ciclovia, zonas).id_busqueda.unique()
        if len(step1)>0:
            return normal(talleres, zonas.iloc[step1, :])
        else:
            print('Lo siento, este tramo de ciclovía no está en la CDMX')        


#%%
# Areas de la ciudad de México 
cdmx = gpd.read_file("data/cdmx.gpkg").to_crs(3857)[['CVE_ENT', 'geometry']]
cl = gpd.read_file('products/ciclovias_centroids_demand.gpkg').to_crs(3857)[['id_centroid','geometry']]
cll = gpd.read_file('products/ciclovias_lines.gpkg').to_crs(3857)
ta = gpd.read_file('Talleres_Bici.gpkg').to_crs(3857)[['id', 'geometry']]

cdmx_zonas = gpd.GeoDataFrame()
for i, poly in enumerate(ox.utils_geo._quadrat_cut_geometry(cdmx['geometry'][0], quadrat_width=5000)):
    cdmx_zonas = cdmx_zonas.append(gpd.GeoDataFrame(index=[i], crs=3857, geometry=[poly]))

fig, ax = plt.subplots(figsize=(10,10))
cdmx_zonas.boundary.plot(ax=ax, alpha=0.5)
cdmx_zonas.iloc[[62,63,76,77],:].boundary.plot(ax=ax, color='green')
cl.plot(ax=ax, color='black', markersize=1)
ta.plot(ax=ax, color='red', markersize=1)
ax.set_axis_off()
plt.tight_layout()
plt.show()

poly = cdmx_zonas.iloc[[62,63,76,77],:]

s=process_time()
espacial(cl, poly)
print('Tiempo: ', round(process_time() - s, 4))

s=process_time()
normal(cl, poly)
print('Tiempo: ', round(process_time() - s, 4))

s=process_time()
espacial(cl, cdmx_zonas)
print('Tiempo: ', round(process_time() - s, 4))

s=process_time()
normal(cl, cdmx_zonas)
print('Tiempo: ', round(process_time() - s, 4))


# Programa para buscar ciclovias y talleres
ciclovias = cl.iloc[:2,:]
resultado = buscador(ciclovias, ta, cdmx_zonas)

fig, ax = plt.subplots(figsize=(10,10))
cdmx_zonas.boundary.plot(ax=ax, alpha=0.4)
ta.plot(ax=ax, alpha=0.8, markersize=2, color='black')
cdmx_zonas.iloc[resultado.id_busqueda.unique()].boundary.plot(ax=ax, color='green')
resultado.plot(ax=ax, color='black', markersize=2)
cll.iloc[list(ciclovias.index), :].plot(ax=ax, color='red')
ax.set_axis_off()
plt.tight_layout()
plt.show()

ciclovias = cl.iloc[:100,:]
resultado = buscador(ciclovias, ta, cdmx_zonas)

fig, ax = plt.subplots(figsize=(10,10))
cdmx_zonas.boundary.plot(ax=ax, alpha=0.4)
ta.plot(ax=ax, alpha=0.8, markersize=2, color='black')
cdmx_zonas.iloc[resultado.id_busqueda.unique()].boundary.plot(ax=ax, color='green')
resultado.plot(ax=ax, color='black', markersize=2)
cll.iloc[list(ciclovias.index), :].plot(ax=ax, color='red')
ax.set_axis_off()
plt.tight_layout()
plt.show()


s=process_time()
buscador(ciclovias, ta, cdmx_zonas)
print('Tiempo: ', round(process_time() - s, 4))

s=process_time()
buscador(ciclovias, ta, cdmx_zonas, False)
print('Tiempo: ', round(process_time() - s, 4))



#%%
'''
####################
## PARALELIZACION ##
####################
'''




#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from os.path import join

import glob
from pathlib import Path

from subprocess import check_output





# Esto es para que no rompa GDAL por algo que quedó mal registrado cuando lo instalé via micromamba
#os.environ['PROJ_LIB'] = "/home/havb/micromamba/pkgs/proj-9.1.0-h93bde94_0/share/proj"
# vars que requiere rasterio
os.environ['GDAL_DATA'] = check_output('pip show rasterio | grep Location | awk \'{print $NF"/rasterio/gdal_data/"}\'', shell=True).decode().strip()
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

import torch

from rastervision.core.data import ClassConfig
from rastervision.core.data import SemanticSegmentationLabels, SemanticSegmentationDiscreteLabels
from rastervision.core.data import RasterioSource, MinMaxTransformer
from rastervision.core.data.label_store.semantic_segmentation_label_store_config import PolygonVectorOutputConfig

from rastervision.pytorch_learner import SemanticSegmentationLearner
from rastervision.pytorch_learner import SolverConfig
from rastervision.pytorch_learner import SemanticSegmentationLearnerConfig
from rastervision.pytorch_learner import SemanticSegmentationSlidingWindowGeoDataset
from rastervision.pytorch_learner import SemanticSegmentationGeoDataConfig


import albumentations as A

import gc


# ## Download data

drive_folder_name = "GEarth_2023_04_zoom17"
output_folder = "/tmp/"

download_img = False

if download_img:
    import gdown
    drive_url = 'https://drive.google.com/drive/folders/1EoeVKGWdO0MyH7H-a0BWbsTFH60B-L5B?usp=sharing'
    gdown.download_folder(url=drive_url, output=output_folder)


# Ubicación de archivos

raster_path = output_folder + drive_folder_name + '/*.tif'
raster_files = glob.glob(raster_path)


# ## Extracción de predicciones


# VERY IMPORTANT
chip_size = 480

# setup del modelo

class_config = ClassConfig(
    names=['background', 'basural'],
    colors=['lightgray', 'darkred'],
    null_class='background')

class_config.ensure_null_class()

model = torch.hub.load(
    'AdeelH/pytorch-fpn:0.3',
    'make_fpn_resnet',
    name='resnet18',
    fpn_type='panoptic',
    num_classes=len(class_config),
    fpn_channels=128,
    in_channels=3,
    out_size=(chip_size, chip_size),
    pretrained=True)

# data_cfg = SemanticSegmentationGeoDataConfig(
#     class_names=class_config.names,
#     class_colors=class_config.colors,
#     num_workers=4, # increase to use multi-processing
# )

# solver_cfg = SolverConfig(
#     batch_sz=60, # en 6 GB de memoria entran unos 85 tensores de imagenes de 256x256 
#     lr=0.01,,
#     #class_loss_weights=[1., 5.]
# )

# learner_cfg = SemanticSegmentationLearnerConfig(data=data_cfg, solver=solver_cfg)

learner = SemanticSegmentationLearner.from_model_bundle(
    model_bundle_uri='../models/GEarth_zoom17/model-bundle.zip',
    output_dir='../models/GEarth_zoom17/',
    model=model, 
    # cfg=learner_cfg,
    training=False
)


# ## Función para iterar en múltiples imagenes

# Definimos Area of Interest
aoi_uri = "../data/aoi/RMBA_envolvente.geojson"


def generate_prediction_polygons(raster_file, aoi_uri, class_config, learner, save_dir):

    print(f'Processing {raster_file}...')

    img_raster_source = RasterioSource(raster_file, allow_streaming=True, raster_transformers=[MinMaxTransformer()])

    pred_ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
        class_config=class_config,
        image_uri=raster_file,
        image_raster_source_kw=dict(allow_streaming=True),
        aoi_uri=aoi_uri,
        size=chip_size,
        stride=chip_size,
        transform=A.Resize(chip_size, chip_size))

    predictions = learner.predict_dataset(
        pred_ds,
        raw_out=True,
        numpy_out=True,
        predict_kw=dict(out_shape=(chip_size, chip_size)),
        progress_bar=True)

    pred_labels = SemanticSegmentationLabels.from_predictions(
        pred_ds.windows,
        predictions,
        smooth=True,
        extent=pred_ds.scene.extent,
        num_classes=len(class_config))

    # liberamos memoria
    del pred_ds
    gc.collect()

    print(f'Prediction completed, writing results to disk')
    pred_labels.save(
        uri=save_dir,
        crs_transformer=img_raster_source.crs_transformer,
        class_config=class_config,
        # set to False to skip writing `labels.tif`
        discrete_output=False,
        # set to False to skip writing `scores.tif`
        smooth_output=False,
        # set to True to quantize floating point score values to uint8 in scores.tif to reduce file size
        smooth_as_uint8=True,
        vector_outputs=[
            # Esto si queremos predicción multiclase
            # PolygonVectorOutputConfig(class_id=i, uri=join(save_dir, f'class-{i}-{mosaic_id}.json'))
            # for i in range(len(class_config))
            # Para una sola clase, ignoramos el background (class_id == 0) y vamos directo a la clase 1
            PolygonVectorOutputConfig(class_id=1, uri=join(save_dir, f'{Path(raster_file).stem}_prediction.json'))
        ]
    )

    # limpieza final

    del predictions
    del pred_labels
    gc.collect()
    print(f'Done: {raster_file}\n')


# ## A predecir

save_dir = "../predicciones/"

for raster_file in raster_files:
    generate_prediction_polygons(raster_file, aoi_uri, class_config, learner, save_dir)


# combinar y disolver los resultados parciales
import geopandas as gpd

partial_results = [join(save_dir, f'{Path(raster_file).stem}_prediction.json') for raster_file in raster_files]

results = gpd.GeoDataFrame()

for file in partial_results:
    results = gpd.GeoDataFrame(gpd.pd.concat([results, gpd.read_file(file)]))

results.to_file(join(save_dir, 'prediction_all.geojson'), driver="GeoJSON")


#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
from subprocess import check_output

# Esto es para que no rompa GDAL por algo que quedó mal registrado cuando lo instalé via micromamba
#os.environ['PROJ_LIB'] = "/home/havb/micromamba/pkgs/proj-9.1.0-h93bde94_0/share/proj"


# vars que requiere rasterio
os.environ['GDAL_DATA'] = check_output('pip show rasterio | grep Location | awk \'{print $NF"/rasterio/gdal_data/"}\'', shell=True).decode().strip()
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'


# In[27]:


import glob

from rastervision.core.data import RasterioSource, MinMaxTransformer

from rastervision.core.data import (
    ClassConfig, GeoJSONVectorSource, RasterioCRSTransformer,
    RasterizedSource, ClassInferenceTransformer)

from rastervision.core.data import SemanticSegmentationLabelSource

from rastervision.core.data.utils.geojson import get_polygons_from_uris
from shapely.geometry import Polygon

from rastervision.pytorch_learner import (
    SemanticSegmentationRandomWindowGeoDataset, SemanticSegmentationSlidingWindowGeoDataset, SemanticSegmentationVisualizer)

import albumentations as A

import torch
from torch.utils.data import ConcatDataset

from rastervision.pytorch_learner import SemanticSegmentationGeoDataConfig
from rastervision.pytorch_learner import SolverConfig
from rastervision.pytorch_learner import SemanticSegmentationLearnerConfig
from rastervision.pytorch_learner import SemanticSegmentationLearner


# In[28]:


## define datasets
chip_size = 160

class_config = ClassConfig(
    names=['background', 'basural'],
    colors=['lightgray', 'darkred'],
    null_class='background')

data_augmentation_transform = A.Compose([
    A.Flip(),
    A.ShiftScaleRotate(),
    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=10),
        A.RandomBrightness(),
        A.RandomGamma(),
    ]),
    A.CoarseDropout(max_height=24, max_width=24, max_holes=4)
])


# In[29]:


## Dataset A
raster_path = '/tmp/GEE/*_2021-*.tif'
raster_files = glob.glob(raster_path)
label_uri = "../../labels/labels_dic_2021.geojson"
aoi_uri = "../../labels/aoi_ALL_buffer_1500_m_labels_dic_2021.geojson"

val_ds_a = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
    class_config=class_config,
    aoi_uri=aoi_uri,
    image_uri=raster_files,
    label_vector_uri=label_uri,
    label_vector_default_class_id=class_config.get_class_id('basural'),
    image_raster_source_kw=dict(allow_streaming=True, raster_transformers=[MinMaxTransformer()]),
    size=chip_size,
    stride=chip_size,
    transform=A.Resize(chip_size, chip_size))


train_ds_a = SemanticSegmentationRandomWindowGeoDataset.from_uris(
    class_config=class_config,
    aoi_uri=aoi_uri,
    image_uri=raster_files,
    label_vector_uri=label_uri,
    label_vector_default_class_id=class_config.get_class_id('basural'),
    image_raster_source_kw=dict(allow_streaming=True, raster_transformers=[MinMaxTransformer()]),
    # window sizes will randomly vary from 100x100 to 300x300
    #size_lims=(100, 300),
    # fixed window size
    size_lims=(chip_size, chip_size+1),
    # resize chips before returning
    out_size=chip_size,
    # allow windows to overflow the extent by 100 pixels
    padding=100,
    max_windows=len(val_ds_a) * 20,  # Atento acá, que tenga el dataset correcto
    transform=data_augmentation_transform
)


# In[ ]:


## Dataset B

raster_path = '/tmp/GEE/*_2017-*.tif'
raster_files = glob.glob(raster_path)
label_uri = "../../labels/labels_nov_2017.geojson"
aoi_uri = "../../labels/aoi_buffer_1500_m_labels_nov_2017.geojson"


val_ds_b = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
    class_config=class_config,
    aoi_uri=aoi_uri,
    image_uri=raster_files,
    label_vector_uri=label_uri,
    label_vector_default_class_id=class_config.get_class_id('basural'),
    image_raster_source_kw=dict(allow_streaming=True, raster_transformers=[MinMaxTransformer()]),
    size=chip_size,
    stride=chip_size,
    transform=A.Resize(chip_size, chip_size))


train_ds_b = SemanticSegmentationRandomWindowGeoDataset.from_uris(
    class_config=class_config,
    aoi_uri=aoi_uri,
    image_uri=raster_files,
    label_vector_uri=label_uri,
    label_vector_default_class_id=class_config.get_class_id('basural'),
    image_raster_source_kw=dict(allow_streaming=True, raster_transformers=[MinMaxTransformer()]),
    # window sizes will randomly vary from 100x100 to 300x300
    #size_lims=(100, 300),
    # fixed window size
    size_lims=(chip_size, chip_size+1),
    # resize chips before returning
    out_size=chip_size,
    # allow windows to overflow the extent by 100 pixels
    padding=100,
    max_windows=len(val_ds_b) * 20,  # Atento acá, que tenga el dataset correcto
    transform=data_augmentation_transform
)


# In[30]:


val_ds_full  = ConcatDataset([val_ds_a, val_ds_b])
train_ds_full  = ConcatDataset([train_ds_a, train_ds_b])



# ### Training a model

# In[36]:


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


# ##### Configure the training

# In[37]:


# data_cfg = SemanticSegmentationGeoDataConfig(
#    class_names=class_config.names,
#    class_colors=class_config.colors,
#    num_workers=4,  # increase to use multi-processing
# )


# In[38]:


# solver_cfg = SolverConfig(
#     batch_sz=450,  # en 6 GB de memoria entran unos 74 tensores de imagenes de 256x256 
#     lr=0.01,
#     # class_loss_weights=[1., 5.]
# )


# In[39]:


# learner_cfg = SemanticSegmentationLearnerConfig(data=data_cfg, solver=solver_cfg)


# #### Initialize Learner

# In[40]:

# learner = SemanticSegmentationLearner(
#     cfg=learner_cfg,
#     output_dir='../models/doble_dataset_AMBA/',
#     model=model,
#     train_ds=train_ds_full,
#     valid_ds=val_ds_full,
# )



learner = SemanticSegmentationLearner.from_model_bundle(
    model_bundle_uri='../models/doble_dataset_AMBA_hybrid_AOI/model-bundle.zip',
    output_dir='../models/doble_dataset_AMBA_hybrid_AOI/',
    model=model,
    train_ds=train_ds_full,
    valid_ds=val_ds_full,
    training=True,
)


# #### Run Tensorboard for monitoring

# In[41]:


#%load_ext tensorboard


# In[42]:


#%tensorboard --bind_all --logdir "/tmp/train-demo/tb-logs" --reload_interval 10


# #### Train – Learner.train()

# In[ ]:


learner.train(epochs=20)


# In[ ]:


learner.save_model_bundle()


# In[ ]:





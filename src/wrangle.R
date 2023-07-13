library(dplyr)
library(sf)
library(rmapshaper)
library(leaflet)


## RELEVAMIENTO 2021
# para el algoritmo hemos estado usando EPSG:32721
relevamiento <- read_sf("~/Downloads/basurales_amba/INFORME 2021/ANALISIS DE PREDICCIONES CIM/CIM ANALISIS PREDICCIONES.shp") %>% 
  st_transform(32721)

## wrangle

# arreglar errores de carga
relevamiento <- relevamiento %>% 
  mutate(Subanalisi = ifelse(Subanalisi == "MASCRA", "MASCARA", Subanalisi)) 


# reflejar cambios de opinión 

dudosos_a_positivos <- c(10134, 7489) # estos se retiran porque los tenemos redibujados en "dudosos_corregidos"
dudosos_a_falsos <- c(9513, 10476, 10156, 8854, 5180) # Estos pasan a "FALSO POSITIVO"

dudosos_corregidos <- read_sf("~/Downloads/basurales_amba/INFORME 2021/ANALISIS DE PREDICCIONES CIM/correcciones/SHAPE DUDOSO 10134.shp") %>% 
  rbind(read_sf("~/Downloads/basurales_amba/INFORME 2021/ANALISIS DE PREDICCIONES CIM/correcciones/SHAPE DUDOSO 7489.shp"))%>% 
  st_transform(32721) %>% 
  transmute(fid = id, DN = NA, Analisis = "POSITIVO", Subanalisi = "NUEVO", 
         Partido = NA, Comentario = NA, geometry)


relevamiento <- relevamiento %>% 
  filter(!(fid %in% dudosos_a_positivos)) %>% 
  mutate(Analisis = ifelse(fid %in% dudosos_a_falsos, "FALSO POSITIVO", Analisis)) %>% 
  rbind(dudosos_corregidos)

relevamiento %>% st_drop_geometry() %>% count(Analisis, Subanalisi)

# transformar para uso con satrpoc/unetseg


# del relevamiento inicial retiramos los "MASCARA" porque ya fueron utilizados para entrenamiento
relevamiento %>% 
  filter(Subanalisi != "MASCARA" | Analisis == "FALSO POSITIVO") %>% 
  transmute(clase = ifelse(Analisis == "FALSO POSITIVO", "no basural", "basural")) %>% 
  write_sf("~/Downloads/basurales_amba/data/DIC-2021/IMAG-SENT-DIC-2021.gpkg", delete_dsn = TRUE)
  

read_sf("~/Downloads/basurales_amba/data/DIC-2021/IMAG-SENT-DIC-2021.gpkg") %>% 
  count(clase)



## RELEVAMIENTO 2017
# para el algoritmo hemos estado usando EPSG:32721
relevamiento <- read_sf("~/Downloads/basurales_amba/INFORME 2017/ANALISIS DE PREDICCIONES CIM/predicciones_nov_2017_rmba_analisis_CIM.shp") %>% 
  st_transform(32721)

relevamiento %>% st_drop_geometry() %>% count(Analisis, Subanalisi)


## wrangle


relevamiento <- relevamiento %>% 
  filter(!(fid %in% dudosos_a_positivos)) %>% 
  mutate(Analisis = ifelse(fid %in% dudosos_a_falsos, "FALSO POSITIVO", Analisis)) 

# transformar para uso con satrpoc/unetseg


# del relevamiento inicial retiramos los "MASCARA" porque ya fueron utilizados para entrenamiento
# también quitamos los "DUDOSO" de acuerdo a lo charlado con el equipo del Centro de Inv Metropolitana
relevamiento %>% 
  filter(Analisis == "FALSO POSITIVO" | Subanalisi %in% c("BASE", "NUEVO")) %>%
  transmute(clase = ifelse(Analisis == "FALSO POSITIVO", "no basural", "basural")) %>% 
  write_sf("~/Downloads/basurales_amba/data/NOV-2017/IMAG-SENT-NOV-2017.gpkg", delete_dsn = TRUE)


read_sf("~/Downloads/basurales_amba/data/NOV-2017/IMAG-SENT-NOV-2017.gpkg") %>% 
  st_is_valid() %>% 
  all


## Consolidar polígonos dic 2021, 
# todos los basurales conocidos + todos los polígonos de falsos positivos



positivos <-read_sf("~/Downloads/139 MASCARAS DICIEMBRE 2021-20230405T182949Z-001/139 MASCARAS DICIEMBRE 2021/139 MASCARAS DICIEMBRE 2021.shp") |> 
  # para el algoritmo hemos estado usando EPSG:32721
  st_transform(32721) |> 
  #por las dudas convertimos multiplys en polys simples
  st_cast("POLYGON") |> 
  transmute(clase = "basural")


falsos_p <- read_sf("~/Downloads/IMAG-SENT-DIC-2021.gpkg") |>
  rename(geometry = geom) |> 
  bind_rows(
    st_transform(
      read_sf("~/Downloads/Predicciones 2021 bis/predicciones_dic_2021_rmba_analisis_CIM.shp"),
      32721)) |> 
  bind_rows(
    st_transform(
      read_sf("~/Downloads/ANALISIS PREDCCIONES CIM/CIM ANALISIS PREDICCIONES.shp"),
      32721)
  ) |> 
  filter(clase == "no basural" | Analisis == "FALSO POSITIVO") |> 
  transmute(clase = "no basural")

# Retiramos de la capa de falsos positivos unos pedacitos que si corresponden a basurales
# (son pisados por algún polígono clasificado como "basural")
# usamos mapshaper siguiendo un consejo aquí https://gis.stackexchange.com/questions/240259/using-simple-features-sf-in-r-how-do-i-erase-polygons-overlapping-with-anothe

falsos_p <- ms_erase(target = falsos_p, erase = positivos) 


positivos |> 
  bind_rows(falsos_p) |> 
  write_sf("/tmp/IMAG-SENT-DIC-2021.gpkg") 

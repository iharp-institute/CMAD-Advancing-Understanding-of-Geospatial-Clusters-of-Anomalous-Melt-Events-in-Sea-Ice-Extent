# CMAD implementation


| **Datasets**                                            | **Resolution/Pixels/Data Size**                                           | **Usage**                                    | **Python Files**                    | **Downloading the Data Source**                  |
|---------------------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------|-------------------------------------|-------------------------------------------------|
| Sea Ice Extent Images (satellite images)                | 332 × 316 pixels (648 MB)                                                  | Detection using CMAD                         | `sea_ice_detection.py`              | [NOAA Sea Ice Extent Images](https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/images/)  |

| Annual Minimum Antarctic Sea Ice Extent|||[Understanding Climate: Antarctic Sea Ice Extent](https://www.climate.gov/news-features/understanding-climate/understanding-climate-antarctic-sea-ice-extent)|
| Aerosol Atmospheric Rivers (AAR) Data                   | Resolution (H=576, W=361, T= 6 hours, 24 years (1997-2020, 677 GB)         | Verification using CMAD_{Benchmark}             | `aerosol_river_verification.py`     | [Atmospheric Rivers Dataset](https://dataverse.ucla.edu/dataset.xhtml?persistentId=doi:10.25346/S6/CXO9PD)                       |
| Integrated Aerosol Transport by AARs                    | Resolution (H=576, W=361, T= 6 hours, 24 years (1997-2020, 697 GB)         | Detection using CMADBenchmark                | `aerosol_transport_detection.py`    | MERRA-2 Dataset                                  |
                   
| Sea Ice Concentration                                   | 1 day                                                                      | Measurement of Anomalous Melting Identified by CMAD | `sea_ice_concentration_analysis.py` | NSIDC Data Archives                             |

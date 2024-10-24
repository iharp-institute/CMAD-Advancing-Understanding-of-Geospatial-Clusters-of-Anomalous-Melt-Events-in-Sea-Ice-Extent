## Convolution Matrix Anomaly Detection (CMAD)

Traditional statistical analyses struggle to reveal the spatial and temporal clusters of anomalous events that contribute significantly to sea ice loss. To overcome this limitation, we introduce a novel method called **Convolution Matrix Anomaly Detection (CMAD)**.

### Key Features of CMAD:

- **No Neural Network Required:** CMAD provides an alternative to Convolutional Neural Networks (CNN) by using an inverse max pooling concept in its convolution operation. This approach is designed specifically for detecting loss (negative values) in sea ice extent, where the traditional CNN's convolution operation fails.
  
- **No Training or Testing Phase:** Unlike traditional CNNs, CMAD doesn't rely on extensive training or testing processes, making it computationally efficient and easy to apply.

- **Satellite Image Utilization:** CMAD utilizes satellite images to monitor and detect loss in Antarctic sea ice extent, identifying clusters of anomalous melting events.

- **Spatiotemporal Detection:** CMAD excels at detecting clusters of contiguous grids representing anomalous events, providing a more comprehensive understanding of sea ice melting patterns.

- **Application Beyond Sea Ice:** The method has the potential to be adapted for other domains where the detection of clustered anomalous events is critical.

### Performance:

CMAD has demonstrated superior capabilities in detecting extreme events, with **87% accuracy** in benchmark data. When compared to traditional methods such as DBSCAN, HDBSCAN, K-Means, Bisecting K-Means, BIRCH, Agglomerative Clustering, OPTICS, and Gaussian Mixtures, **CMAD$_{Benchmark}$** exhibits heightened sensitivity and efficacy in capturing significant variations in dynamic multidimensional datasets.

### Application:

Our analysis shows that CMAD effectively detected clusters of anomalous melting patterns affecting the Weddell and Ross Sea regions in Antarctica. The anomalous melting process was first observed along the outer boundary of the sea ice extent in September 2022 and gradually spread throughout the region by February 2023.

### Comparative Analysis:

CMAD$_{Benchmark}$ outperformed traditional clustering and anomaly detection methods, proving its value as a more accurate and efficient tool for detecting extreme events in multidimensional datasets over time.

---

To run the code and replicate our results, follow the installation instructions and refer to the provided notebooks, including the `optics.ipynb` file, for examples of clustering algorithms applied to the AAR dataset. Make sure to install all necessary dependencies before running the code.





| **Datasets**                                            | **Resolution/Pixels/Data Size**                                           | **Usage**                                    | **Python or Jupyter Files Files**                    | **Downloading the Data Source**                  |
|---------------------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------|-------------------------------------|-------------------------------------------------|
| Sea Ice Extent Images (satellite images)                | 332 × 316 pixels (648 MB)                                                  | Detection using CMAD                         | Image_processing.ipynb, Image_processing_for_py_gpu.ipynb, discord_km_2__only_7_days.ipynb, discord_km_2__only_7_days_for_thesis.ipynb          | [NOAA Sea Ice Extent Images](https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/images/)  |
| Annual Minimum Antarctic Sea Ice Extent                 | N/A                                                                       | Visualization and trend analysis             |discord_km_2__only_7_days.ipynb, discord_km_2__only_7_days_for_thesis.ipynb                | [Understanding Climate: Antarctic Sea Ice Extent](https://www.climate.gov/news-features/understanding-climate/understanding-climate-antarctic-sea-ice-extent) |
| Aerosol Atmospheric Rivers (AAR) Data                   | Resolution (H=576, W=361, T= 6 hours, 24 years (1997-2020, 677 GB)         | Detection using CMAD_Benchmark             | cmad_AAR.ipynb,cmad_AAR_for_gpu.ipynb    | [Atmospheric Rivers Dataset](https://dataverse.ucla.edu/dataset.xhtml?persistentId=doi:10.25346/S6/CXO9PD)|
| Integrated Aerosol Transport by AARs                    | Resolution (H=576, W=361, T= 6 hours, 24 years (1997-2020, 697 GB)         | Verification using CMAD_Benchmark                | data_cmad.ipynb   | [MERRA-2 Dataset Processed](https://acp.copernicus.org/articles/22/8175/2022/)                                |
| Sea Ice Concentration                                   | 1 day                                                                      | Measurement of Anomalous Melting Identified by CMAD | discord_km_2__only_7_days.ipynb, discord_km_2__only_7_days_for_thesis.ipynb | [NSIDC Data](https://nsidc.org/data/nsidc-0051/versions/2)                             |





## Benchmarking with AAR Data Using Clustering Algorithms from scikit-learn

For benchmarking with AAR data, I applied all the clustering algorithms available in the `scikit-learn` library. Readers can use these algorithms by importing the `scikit-learn` library. A demo has been provided using the **OPTICS** algorithm in the `optics.ipynb` file.

### Notebook:
- `optics.ipynb`

### Clustering Algorithms:
- Available in `scikit-learn` (e.g., OPTICS, K-Means, DBSCAN, Agglomerative, etc.)

### Instructions:
Readers can follow the example in `optics.ipynb` to apply other clustering algorithms by modifying the code. Simply import the clustering algorithm in the notebook with the desired method from [scikit-learn](https://scikit-learn.org/1.5/modules/clustering.html).

### Dependencies:
Before running the code, make sure to install the necessary libraries.



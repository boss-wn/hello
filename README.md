# KPI-Survey


A curated list of awesome publications on KPI analysis.   
  
Table of Contents    

- Awesome KPI Analysis Research  
	- [Conferences and Journals](#1)
	- [Research Groups](#2)  
    - [Anomaly Detection](#3)
        - [Univariate KPI](#4)
            - [Supervised Learning](#5)
            - [Unsupervised Learning](#6) 
            - [Semi-Supervised Learning](#7)
        - [Multivariate KPIs](#8)
        - [Multi-Dimension](#9)
    - [Clustering](#10)
 	- [Correlation Analysis](#11)
	- [Extension](#12)	
    - [Datasets](#13)  
    - [Open Source Code](#14)
	- [Credits](#15)
	- [License](#16)  


## <span id="1">Conferences and Journals</span> 
   
Key Performance Indicators (KPIs) are a type of valuable data generated from many sources such as software, systems, networks, devices, etc.     
They have also been used for a number of tasks related to reliability, security, performance, and energy. Therefore, the research of KPI analysis has attracted interests from different research areas.


- System area    
	- Conferences: [(OSDI)](https://dblp.uni-trier.de/db/conf/osdi/index.html) | [(SOSP)](https://dblp.uni-trier.de/db/conf/sosp/index.html) | [ATC](https://dblp.uni-trier.de/db/conf/atc/index.html) | [SIGMETRICS](https://dblp.uni-trier.de/db/conf/sigmetrics/index.html) | [Middleware](https://dblp.uni-trier.de/db/conf/middleware/index.html) | [(ICDCS)](https://dblp.uni-trier.de/db/conf/icdcs/index.html) 
	- Journals: [(TC)](https://dblp.uni-trier.de/db/journals/tc/index.html) | [(TOCS)](https://dblp.uni-trier.de/db/journals/tocs/index.html) | [(TPDS)](https://dblp.uni-trier.de/db/journals/tpds/index.html)  
- Cloud computing area  
	- Conferences: [(SoCC)](https://dblp.uni-trier.de/db/conf/cloud/index.html) | [(CLOUD)](https://dblp.uni-trier.de/db/conf/IEEEcloud/index.html)
	- Journals: [(TCC)](https://dblp.uni-trier.de/db/journals/tcc/index.html)
- Networking area    
	- Conferences: [(NSDI)](https://dblp.uni-trier.de/db/conf/nsdi/index.html) | [INFOCOMM](https://dblp.uni-trier.de/db/conf/infocom/index.html) | [IMC](http://www.sigcomm.org/events/imc-conference) | [IWQoS](https://dblp.uni-trier.de/db/conf/iwqos/index.html) | [IPCCC](https://dblp.uni-trier.de/db/conf/ipccc/index.html) | [MobiCom](https://dblp.uni-trier.de/db/conf/mobicom/index.html) | [CoNEXT](https://dblp.uni-trier.de/db/conf/conext/index.html) 
	- Journals: [(TON)](https://dblp.uni-trier.de/db/journals/ton/index.html) | [IEEE TNSM](https://dblp.uni-trier.de/db/journals/tnsm/index.html) 
- Software engineering area    
	- Conferences: [(ICSE)](https://dblp.uni-trier.de/db/conf/icse/index.html) | [(FSE)](https://dblp.uni-trier.de/db/conf/fse/index.html) | [ASE](https://dblp.uni-trier.de/db/conf/kbse/index.html)  
	- Journals: [(TSE)](https://dblp.uni-trier.de/db/journals/tse/index.html) | [(TOSEM)](https://dblp.uni-trier.de/db/journals/tosem/index.html)  
- Reliability area  
	- Conferences: [(DSN)](https://dblp.uni-trier.de/db/conf/dsn/index.html) | [ISSRE](https://dblp.uni-trier.de/db/conf/issre/index.html) | [(SRDS)](https://dblp.uni-trier.de/db/conf/srds/index.html)  
	- Journals: [(TDSC)](https://dblp.uni-trier.de/db/journals/tdsc/index.html) | [(TR)](https://dblp.uni-trier.de/db/journals/tr/index.html)
- Security area    
	- Conferences: [CCS](http://www.wikicfp.com/cfp/servlet/event.showcfp?eventid=83847) | [DSN](http://dsn2021.ntu.edu.tw/) 
	- Journals: [TDSC](https://dblp.uni-trier.de/db/journals/tdsc/index.html)
- AI area     
	- Conferences: [KDD](https://dblp.uni-trier.de/db/conf/kdd/index.html) | [NeurIPS] | [AAAI](https://dblp.uni-trier.de/db/conf/aaai/index.html) | [(CIKM)](https://dblp.uni-trier.de/db/conf/cikm/index.html) | [(ICDM)](https://dblp.uni-trier.de/db/conf/icdm/index.html) | [BigData](https://dblp.uni-trier.de/db/conf/bigdata/index.html)
	- Journals: [(TKDE)](https://dblp.uni-trier.de/db/journals/tkde/index.html) | [(TBD)](https://dblp.uni-trier.de/db/journals/tbd/index.html) 
- Database area  
	- Conferences: [VLDB](https://dblp.uni-trier.de/db/conf/vldb/index.html) | [SIGMOD](https://dblp.uni-trier.de/db/conf/sigmod/index.html) 
	- Journals: [Pattern](https://dblp.uni-trier.de/db/journals/pr/index.html)
- Industrial area    
	- Conferences: [(SREcon)](https://www.usenix.org/conferences/byname/925) | [(GOPS)](https://www.bagevent.com/event/GOPS2019-shenzhen?bag_track=bagevent)  
  
## <span id="2">Research Groups</span>   
 
|    China (& HK)   |              |               |
| -------------     | ------------ | ------------- |  
| [Dan Pei](https://netman.aiops.org/~peidan/), Tsinghua  | [Shenglin Zhang](http://nkcs.iops.ai/shenglinzhang/), Nankai  |[Shaoxu Song](http://ise.thss.tsinghua.edu.cn/sxsong/), Tsinghua    
  

## <span id="3">Anomaly Detection</span>

### <span id="4">Univariate KPI</span>

#### <span id="5">Supervised Learning</span>

- [KDD 2018] [Deep r-th Root of Rank Supervised Joint Binary Embedding for Multivariate Time Series Retrieval](https://doi.org/10.1145/3219819.3220108)
- [IMC 2015] [Opprentice: Towards Practical and Automatic Anomaly Detection Through Machine Learning](http://dx.doi.org/10.1145/2815675.2815679.)  
- [KDD 2015] [Learning a Hierarchical Monitoring System for Detecting and Diagnosing Service Issues](http://dx.doi.org/10.1145/2783258.2788624)  
- [KDD 2015] [Generic and Scalable Framework for Automated Time-series Anomaly Detection](http://dx.doi.org/10.1145/2783258.2788611)

#### <span id="6">Unsupervised Learning</span>

- [AAAI 2020] MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams 
- [NeurIPS 2020] Timeseries Anomaly Detection using Temporal Hierarchical One-Class Network
- [VLDB 2020] [Diagnosing Root Causes of Intermittent Slow Queries in Cloud Databases]( https://doi.org/10.14778/3389133.3389136)
- [IWQoS 2020] [Localizing Failure Root Causes in a Microservice through Causality Inference](https://ieeexplore.ieee.org/document/9213058)  
- [IJCAI 2019] BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time
- [KDD 2019] [Time-Series Anomaly Detection Service at Microsoft](https://doi.org/10.1145/3292500.3330680)  
- [INFOCOM 2019] [Unsupervised Anomaly Detection for Intricate KPIs via Adversarial Training of VAE](https://ieeexplore.ieee.org/document/8737430)  
- [TNSM 2019] [Unsupervised Online Anomaly Detection with Parameter Adaptation for KPI Abrupt Changes](https://ieeexplore.ieee.org/document/8944284)  
- [CCS 2018] Truth Will Out: Departure-Based Process-Level Detection of Stealthy Attacks on Control Systems
- [WWW 2018] [Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications]( https://doi.org/10.1145/3178876.3185996)  
- [IPCCC 2018] [Robust and Unsupervised KPI Anomaly Detection Based on Conditional Variational Autoencoder](https://ieeexplore.ieee.org/document/8710885)  
- [KDD 2017] [Anomaly Detection in Streams with Extreme Value Theory](https://doi.org/10.1145/3097983.3098144)  
- [SIGMOD 2016] [DBSherlock: A Performance Diagnostic Tool for Transactional Databases](http://dx.doi.org/10.1145/2882903.2915218)
              
#### <span id="7">Semi-Supervised Learning</span> 
  
- [KDD 2019] [An Adaptive Approach for Anomaly Detector Selection and Fine-Tuning in Time Series](https://doi.org/10.1145/3326937.3341253)
- [ATC 2019] [Cross-dataset Time Series Anomaly Detection for Cloud Systems](https://www.usenix.org/conference/atc19/presentation/zhang-xu)    
- [BigData 2019] [Intelligent Detection of Large-Scale KPI Streams Anomaly Based on Transfer Learning](https://doi.org/10.1007/978-981-15-1899-7_26)
- [KDD 2018] [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://doi.org/10.1145/3219819.3219845) 
- [IPCCC 2018] [Rapid Deployment of Anomaly Detection Models for Large Number of Emerging KPI Streams](https://ieeexplore.ieee.org/document/8711315/)    
- [PATTERN 2018] [Semi-supervised time series classification on positive and unlabeled problems using cross-recurrence quantification analysis](https://doi.org/10.1016/j.patcog.2018.02.030)  
 
### <span id="8">Multivariate KPIs</span>

- [ASE 2020] [Jump-Starting Multivariate Time Series Anomaly Detection for Online Service Systems]() 
- [KDD 2020] [USAD: UnSupervised Anomaly Detection on Multivariate Time](https://doi.org/10.1145/3394486.3403392)
- [KDD 2019] [Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network](https://doi.org/10.1145/3292500.3330672)   
- [AAAI 2019] [A Deep Neural Network for Unsupervised Anomaly  Detection and Diagnosis in Multivariate Time Series Data](https://www.aaai.org/ojs/index.php/AAAI/article/view/3942)

### <span id="9">Multi-Dimension</span>

- [INFOCOM 2019] [Detecting Anomaly in Large-scale Network using Mobile Crowdsourcing](https://ieeexplore.ieee.org/document/8737541/) 
- [MobiCom 2015] [ABSENCE: Usage-based Failure Detection in Mobile Networks](http://dx.doi.org/10.1145/2789168.2790127)  
- [SIGMETRICS 2015] [Detecting and Localizing End-to-End Performance Degradation for Cellular Data Services](https://dl.acm.org/doi/10.1145/2745844.2745892) 

			
## <span id="10">Clustering</span>

- [KDD 2018] [Deep r-th Root of Rank Supervised Joint Binary Embedding for Multivariate Time Series Retrieval](https://dl.acm.org/doi/10.1145/3219819.3220108)
- [IWQoS 2018] [Robust and Rapid Clustering of KPIs for Large-Scale Anomaly Detection](https://ieeexplore.ieee.org/document/8624168)
- [Middleware 2017] [Sieve: Actionable Insights from Monitored Metrics in Distributed Systems](https://dl.acm.org/doi/10.1145/3135974.3135977)
- [VLDB 2015] [YADING: Fast Clustering of Large-Scale Time Series Data](https://dl.acm.org/doi/10.14778/2735479.2735481)

### <span id="11">Correlation Analysis</span>

- [IWQoS 2019] [CoFlux: Robustly Correlating KPIs by Fluctuations for Service Troubleshooting](https://doi.org/10.1145/3326285.3329048)  
- [KDD 2014] [Correlating Events with Time Series for Incident Diagnosis](http://dx.doi.org/10.1145/2623330.2623374) 
 
### <span id="12">Extension</span>

- [IEEE TNSM 2019] [Automatic and Generic Periodicity Adaptation for KPI Anomaly Detection](https://ieeexplore.ieee.org/document/8723601)
- [INFOCOM 2019] [Label-Less: A Semi-Automatic Labelling Tool for KPI Anomalies](https://ieeexplore.ieee.org/document/8737429) 
- [ISSRE 2018] [Robust and Rapid Adaption for Concept Drift in Software System Anomaly Detection](https://ieeexplore.ieee.org/document/8539065)
- [CoNEXT 2015] [Rapid and Robust Impact Assessment of Software Changes in Large Internet-based Services]( http://dx.doi.org/10.1145/2716281.2836087)
 
## <span id="13">Datasets</span>

[Univariate KPI anomaly detection] AIOps Challenge 2018 Datasets: [https://github.com/NetManAIOps/KPI-Anomaly-Detection](https://github.com/NetManAIOps/KPI-Anomaly-Detection)     
[Univariate KPI anomaly detection] Bagel Datasets: [https://github.com/NetManAIOps/Bagel](https://github.com/NetManAIOps/Bagel)     
[Univariate KPI anomaly detection] Dount Datasets: [https://github.com/NetManAIOps/donut](https://github.com/NetManAIOps/donut)  
[Univariate KPI anomaly detection] NAB Datasets: [https://github.com/numenta/NAB](https://github.com/numenta/NAB)   
[Univariate KPI anomaly detection] Yahoo's Webscope S5: [https://github.com/waico/SkAB](https://github.com/waico/SkAB)    
[Multivariate KPIs anomaly detection] OmniAnomaly Datasets: [https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)       
[Multi-dimension KPIs hotspot localization] AIOps Challenge 2019 Datasets: [https://github.com/NetManAIOps/MultiDimension-Localization](https://github.com/NetManAIOps/MultiDimension-Localization)     
[Multi-dimension KPIs hotspot localization] Squeeze Datasets: [https://github.com/NetManAIOps/Squeeze](https://github.com/NetManAIOps/Squeeze)  
[Multivariate KPIs anomaly detection] SKAB Datasets: [https://github.com/waico/SkAB](https://github.com/waico/SkAB)         
[Trace anomaly detection] TraceAnomaly Datasets: [https://github.com/NetManAIOps/TraceAnomaly](https://github.com/NetManAIOps/TraceAnomaly)
 

## <span id="14">Open Source Code</span>

[C++] [MIDAS](https://github.com/bhatiasiddharth/MIDAS): MIDAS, short for Microcluster-Based Detector of Anomalies in Edge Streams, detects microcluster anomalies from an edge stream in constant time and memory.  
[C++] [Nupic](https://github.com/numenta/nupic): Numenta Platform for Intelligent Computing is an implementation of Hierarchical Temporal Memory (HTM).     
[Go] [Anomalyzer](https://github.com/lytics/anomalyzer): Anomalyzer implements a suite of statistical tests that yield the probability that a given set of numeric input, typically a time series, contains anomalous behavior.  
[Go] [banshee](https://github.com/facesea/banshee): Anomalies detection system for periodic metrics.  
[Java] [Surus](https://github.com/Netflix/Surus): A collection of tools for analysis in Pig and Hive.   
[Java] [Adaptive Alerting](https://github.com/ExpediaDotCom/adaptive-alerting): Streaming anomaly detection with automated model selection and fitting.  
[Java] [EGADS](https://github.com/yahoo/egads): GADS is a library that contains a number of anomaly detection techniques applicable to many use-cases in a single package with the only dependency being Java.  
[Python] [TODS](https://github.com/datamllab/tods): TODS is a full-stack automated machine learning system for outlier detection on multivariate time-series data.   
[Python] [Skyline](https://github.com/earthgecko/skyline): Skyline is a near real time anomaly detection system.   
[Python] [Banpei](https://github.com/tsurubee/banpei): Banpei is a Python package of the anomaly detection.  
[Python] [Telemanom](https://github.com/khundman/telemanom): A framework for using LSTMs to detect anomalies in multivariate time series data.  
[Python] [DeepADoTS](https://github.com/KDD-OpenSource/DeepADoTS): A benchmarking pipeline for anomaly detection on time series data for multiple state-of-the-art deep learning methods.  
[Python] [NAB](https://github.com/numenta/NAB): The Numenta Anomaly Benchmark: NAB is a novel benchmark for evaluating algorithms for anomaly detection in streaming, real-time applications.  
[Python] [ADTK](https://github.com/arundo/adtk): Anomaly Detection Toolkit (ADTK) is a Python package for unsupervised / rule-based time series anomaly detection.       
[Python] [CAD](https://github.com/smirmik/CAD): Contextual Anomaly Detection for real-time AD on streagming data (winner algorithm of the 2016 NAB competition).   
[Python] [datastream.io](https://github.com/MentatInnovations/datastream.io): n open-source framework for real-time anomaly detection using Python, Elasticsearch and Kibana.   
[Python] [Donut](https://github.com/korepwx/donut): Donut is an unsupervised anomaly detection algorithm for seasonal KPIs, based on Variational Autoencoders.  
[Python] [LoudML](https://github.com/regel/loudml): Loud ML is an open source time series inference engine built on top of TensorFlow. It's useful to forecast data, detect outliers, and automate your process using future knowledge.  
[Python] [luminol](https://github.com/linkedin/luminol): Luminol is a light weight python library for time series data analysis. The two major functionalities it supports are anomaly detection and correlation. It can be used to investigate possible causes of anomaly.  
[Python] [PyOD](https://pyod.readthedocs.io/en/latest/): PyOD is a comprehensive and scalable Python toolkit for detecting outlying objects in multivariate data.  
[Python] [PyOdds](https://github.com/datamllab/pyodds): PyODDS is an end-to end Python system for outlier detection with database support. PyODDS provides outlier detection algorithms, which support both static and time-series data.  
[Pythno] [PySAD](https://github.com/selimfirat/pysad): PySAD is a streaming anomaly detection framework with various online models and complete set of tools for experimentation.  
[Python] [rrcf](https://github.com/kLabUM/rrcf): Implementation of the Robust Random Cut Forest algorithm for anomaly detection on streams.  
[Python] [ruptures](https://github.com/deepcharles/ruptures/): Ruptures is a Python library for off-line change point detection. This package provides methods for the analysis and segmentation of non-stationary signals.  
[Python] [Telemanom](https://github.com/khundman/telemanom): A framework for using LSTMs to detect anomalies in multivariate time series data. Includes spacecraft anomaly data and experiments from the Mars Science Laboratory and SMAP missions.  
[Python] [Luminaire](https://github.com/zillow/luminaire): Luminaire is a python package that provides ML driven anomaly detection and forecasting solutions for time series data.  
[Python] [GluonTS](https://github.com/awslabs/gluon-ts): GluonTS is a Python toolkit for probabilistic time series modeling, built around MXNet. GluonTS provides utilities for loading and iterating over time series datasets, state of the art models ready to be trained, and building blocks to define your own models.  
[Python] [pmdarima](https://github.com/tgsmith61591/pyramid): Porting of R's _auto.arima_ with a scikit-learn-friendly interface.  
[Python/R] [Prophet](https://github.com/facebook/prophet): Prophet is a procedure for forecasting time series data. It is based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays.  
[Python] [PyFlux](https://github.com/RJT1990/pyflux): The library has a good array of modern time series models, as well as a flexible array of inference options (frequentist and Bayesian) that can be applied to these models.  
[Python] [SaxPy](https://github.com/seninp/saxpy): General implementation of SAX, as well as HOTSAX for anomaly detection.  
[Python] [seglearn](https://github.com/dmbee/seglearn): Seglearn is a python package for machine learning time series or sequences. It provides an integrated pipeline for segmentation, feature extraction, feature processing, and final estimator.  
[Python] [Tigramite](https://github.com/jakobrunge/tigramite): Tigramite is a causal time series analysis python package. It allows to efficiently reconstruct causal graphs from high-dimensional time series datasets and model the obtained causal dependencies for causal mediation and prediction analyses.  
[Python] [tslearn](https://github.com/rtavenar/tslearn): tslearn is a Python package that provides machine learning tools for the analysis of time series. This package builds on scikit-learn, numpy and scipy libraries.  
[Python] [Curve](https://github.com/baidu/Curve): Curve is an open-source tool to help label anomalies on time-series data.   
[Python3] [Skyline](https://github.com/earthgecko/skyline): Skyline is a real-time anomaly detection system, built to enable passive monitoring of hundreds of thousands of metrics.  
[Python + node.js] [Hastic](https://github.com/hastic): Anomaly detection tool for time series data with Grafana-based UI.  
[R] [AnomalyDetection](https://github.com/twitter/AnomalyDetection): AnomalyDetection is an open-source R package to detect anomalies which is robust, from a statistical standpoint, in the presence of seasonality and an underlying trend.    
[R] [Anomalize](https://cran.r-project.org/web/packages/anomalize/): The 'anomalize' package enables a "tidy" workflow for detecting anomalies in data.  
[R] [oddstream](https://github.com/pridiltal/oddstream): oddstream (Outlier Detection in Data Streams) provides real time support for early detection of anomalous series within a large collection of streaming time series data.  
[R] [Taganomaly](https://github.com/Microsoft/TagAnomaly): Simple tool for tagging time series data. Works for univariate and multivariate data, provides a reference anomaly prediction using Twitter's AnomalyDetection package.
## <span id="15">Credits</span>
- Anomaly detection [Yzhao062](https://github.com/yzhao062/anomaly-detection-resources)
- Log survey [Logpai](https://github.com/logpai/log-survey)  
- Anomaly detection [Hoya012](https://github.com/hoya012/awesome-anomaly-detection) 
- Awesome-TS-anomaly-detection [rob-med](https://github.com/rob-med/awesome-TS-anomaly-detection)
## <span id="16">License</span>     










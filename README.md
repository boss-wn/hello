# Awesome KPI Analysis Research  


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
    - [Open Source Code](14)
	- [Credits](#15)
	- [License](#16)  
## <span id="1">Conferences and Journals</span> 
Kpis are a type of valuable data generated from many sources such as software, systems, networks, devices, etc.   
They have also been used for a number of tasks related to reliability, security, performance, and energy. Therefore, the research of log analysis has attracted interests from different research areas.    

- System area  
	- Conferences:[(OSDI)](https://dblp.uni-trier.de/db/conf/osdi/index.html) | [(SOSP)](https://dblp.uni-trier.de/db/conf/sosp/index.html) | [ATC](https://dblp.uni-trier.de/db/conf/atc/index.html) | [(ICDCS)](https://dblp.uni-trier.de/db/conf/icdcs/index.html)  
	- Journals:[(TC)](https://dblp.uni-trier.de/db/journals/tc/index.html) | [(TOCS)](https://dblp.uni-trier.de/db/journals/tocs/index.html) | [(TPDS)](https://dblp.uni-trier.de/db/journals/tpds/index.html)  
- Cloud computing area:
	- Conferences:[(SoCC)](https://dblp.uni-trier.de/db/conf/cloud/index.html) | [(CLOUD)](https://dblp.uni-trier.de/db/conf/IEEEcloud/index.html)
	- Journals:[(TCC)](https://dblp.uni-trier.de/db/journals/tcc/index.html)
- Networking area:  
	- Conferences:[(NSDI)](https://dblp.uni-trier.de/db/conf/nsdi/index.html) | [INFOCOMM](https://dblp.uni-trier.de/db/conf/infocom/index.html)  
	- Journals:[(TON)](https://dblp.uni-trier.de/db/journals/ton/index.html)  
- Software engineering area  
	- Conferences:[(ICSE)](https://dblp.uni-trier.de/db/conf/icse/index.html) | [(FSE)](https://dblp.uni-trier.de/db/conf/fse/index.html) | [(ASE)](https://dblp.uni-trier.de/db/conf/kbse/index.html)  
	- Journals:[(TSE)](https://dblp.uni-trier.de/db/journals/tse/index.html) | [(TOSEM)](https://dblp.uni-trier.de/db/journals/tosem/index.html)  
- Reliability area:
	- Conferences:[(DSN)](https://dblp.uni-trier.de/db/conf/dsn/index.html) | [(ISSRE)](https://dblp.uni-trier.de/db/conf/issre/index.html) | [(SRDS)](https://dblp.uni-trier.de/db/conf/srds/index.html)  
	- Journals:[(TDSC)](https://dblp.uni-trier.de/db/journals/tdsc/index.html) | [(TR)](https://dblp.uni-trier.de/db/journals/tr/index.html)
- Security area:  
	- Conferences:[KDD](https://dblp.uni-trier.de/db/conf/kdd/index.html) | [(CIKM)](https://dblp.uni-trier.de/db/conf/cikm/index.html) | [(ICDM)](https://dblp.uni-trier.de/db/conf/icdm/index.html) | [BigData](https://dblp.uni-trier.de/db/conf/bigdata/index.html)
	- Journals:[(TKDE)](https://dblp.uni-trier.de/db/journals/tkde/index.html) | [(TBD)](https://dblp.uni-trier.de/db/journals/tbd/index.html)
- Industrial conferneces:  
	- [(SREcon)](https://www.usenix.org/conferences/byname/925) | [(GOPS)](https://www.bagevent.com/event/GOPS2019-shenzhen?bag_track=bagevent)
  
## <span id="2">Research Groups</span>   
 
|    China (& HK)   |              |               |
| -------------     | ------------ | ------------- |  
| [Dan Pei](https://netman.aiops.org/~peidan/),Tsinghua  | [Shenglin Zhang](http://nkcs.iops.ai/shenglinzhang/),Nankai  |[Shaoxu Song](http://ise.thss.tsinghua.edu.cn/sxsong/),Tsinghua  
|  1   |   1  |   1  |  
  
##<span id="3">Anomaly Detection</span>
###<span id="4">Univariate KPI</span>
####<span id="5">Supervised Learning</span>
- [KDD 2018] [Deep r-th Root of Rank Supervised Joint Binary Embedding for Multivariate Time Series Retrieval](https://doi.org/10.1145/3219819.3220108)
- [IMC 2015] [Opprentice: Towards Practical and Automatic Anomaly Detection Through Machine Learning](http://dx.doi.org/10.1145/2815675.2815679.)  
- [KDD 2015] [Learning a Hierarchical Monitoring System for Detecting and Diagnosing Service Issues](http://dx.doi.org/10.1145/2783258.2788624)  
- [KDD 2015] [Generic and Scalable Framework for Automated Time-series Anomaly Detection](http://dx.doi.org/10.1145/2783258.2788611)
####<span id="6">Unsupervised Learning</span>
- [VLDB 2020] [Diagnosing Root Causes of Intermittent Slow Queries in Cloud Databases]( https://doi.org/10.14778/3389133.3389136)
- [IWQoS 2020] [Localizing Failure Root Causes in a Microservice through Causality Inference](https://ieeexplore.ieee.org/document/9213058)  
- [KDD 2019] [Time-Series Anomaly Detection Service at Microsoft](https://doi.org/10.1145/3292500.3330680)  
- [INFOCOM 2019] [Unsupervised Anomaly Detection for Intricate KPIs via Adversarial Training of VAE](https://ieeexplore.ieee.org/document/8737430)  
- [IEEE TNSM 2019] [Unsupervised Online Anomaly Detection with Parameter Adaptation for KPI Abrupt Changes](https://ieeexplore.ieee.org/document/8944284)  
- [WWW 2018] [Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications]( https://doi.org/10.1145/3178876.3185996)  
- [IPCCC 2018] [Robust and Unsupervised KPI Anomaly Detection Based on Conditional Variational Autoencoder](https://ieeexplore.ieee.org/document/8710885)  
- [KDD 2017] [Anomaly Detection in Streams with Extreme Value Theory](https://doi.org/10.1145/3097983.3098144)  
- [SIGMOD 2016] [DBSherlock: A Performance Diagnostic Tool for Transactional Databases](http://dx.doi.org/10.1145/2882903.2915218)              
####<span id="7">Semi-Supervised Learning</span>   
- [KDD 2019] [An Adaptive Approach for Anomaly Detector Selection and Fine-Tuning in Time Series](https://doi.org/10.1145/3326937.3341253)
- [ATC 2019] [Cross-dataset Time Series Anomaly Detection for Cloud Systems](https://www.usenix.org/conference/atc19/presentation/zhang-xu)    
- [BigData 2019] [Intelligent Detection of Large-Scale KPI Streams Anomaly Based on Transfer Learning](https://doi.org/10.1007/978-981-15-1899-7_26)
- [KDD 2018] [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://doi.org/10.1145/3219819.3219845) 
- [IPCCC 2018] [Rapid Deployment of Anomaly Detection Models for Large Number of Emerging KPI Streams](https://ieeexplore.ieee.org/document/8711315/)    
- [PATTERN 2018] [Semi-supervised time series classification on positive and unlabeled problems using cross-recurrence quantification analysis](https://doi.org/10.1016/j.patcog.2018.02.030)  
###<span id="8">Multivariate KPIs</span>
- [ASE 2020] [Jump-Starting Multivariate Time Series Anomaly Detection for Online Service Systems ]() 
- [KDD 2020] [USAD : UnSupervised Anomaly Detection on Multivariate Time](https://doi.org/10.1145/3394486.3403392)
- [KDD 2019] [Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network](https://doi.org/10.1145/3292500.3330672)   
- [AAAI 2019] [A Deep Neural Network for Unsupervised Anomaly  Detection and Diagnosis in Multivariate Time Series Data](https://www.aaai.org/ojs/index.php/AAAI/article/view/3942) 
###<span id="9">Multi-Dimension</span>
- [INFOCOM 2019] [Detecting Anomaly in Large-scale Network using Mobile Crowdsourcing](https://ieeexplore.ieee.org/document/8737541/) 
- [MobiCom 2015] [ABSENCE: Usage-based Failure Detection in Mobile Networks](http://dx.doi.org/10.1145/2789168.2790127)  
- [SIGMETRICS 2015] [Detecting and Localizing End-to-End Performance Degradation for Cellular Data Services](https://dl.acm.org/doi/10.1145/2745844.2745892) 
##<span id="10">Clustering</span>
- [KDD 2018] [Deep r-th Root of Rank Supervised Joint Binary Embedding for Multivariate Time Series Retrieval](https://dl.acm.org/doi/10.1145/3219819.3220108)
- [IWQoS 2018] [Robust and Rapid Clustering of KPIs for Large-Scale Anomaly Detection](https://ieeexplore.ieee.org/document/8624168)
- [Mid 2017] [Sieve: Actionable Insights from Monitored Metrics in Distributed Systems](https://dl.acm.org/doi/10.1145/3135974.3135977)
- [VLDB 2015] [YADING: Fast Clustering of Large-Scale Time Series Data](https://dl.acm.org/doi/10.14778/2735479.2735481)
###<span id="11">Correlation Analysis</span>
- [IWQoS 2019] [CoFlux: Robustly Correlating KPIs by Fluctuations for Service Troubleshooting](https://doi.org/10.1145/3326285.3329048)  
- [KDD 2014] [Correlating Events with Time Series for Incident Diagnosis](http://dx.doi.org/10.1145/2623330.2623374)  
###<span id="12">Extension</span>
- [IEEE TNSM 2019] [Automatic and Generic Periodicity Adaptation for KPI Anomaly Detection](https://ieeexplore.ieee.org/document/8723601)
- [INFOCOM 2019] [Label-Less: A Semi-Automatic Labelling Tool for KPI Anomalies](https://ieeexplore.ieee.org/document/8737429) 
- [ISSRE 2018] [Robust and Rapid Adaption for Concept Drift in Software System Anomaly Detection](https://ieeexplore.ieee.org/document/8539065)
- [CoNEXT 2015] [Rapid and Robust Impact Assessment of Software Changes in Large Internet-based Services]( http://dx.doi.org/10.1145/2716281.2836087) 
##<span id="13">Datasets</span>
AIOps Challenge 2020 Datasets:[https://github.com/NetManAIOps/AIOps-Challenge-2020-Data](https://github.com/NetManAIOps/AIOps-Challenge-2020-Data)  
AIOps Challenge 2019 Datasets:[https://github.com/NetManAIOps/MultiDimension-Localization](https://github.com/NetManAIOps/MultiDimension-Localization)  
AIOps Challenge 2018 Datasets:[https://github.com/NetManAIOps/KPI-Anomaly-Detection](https://github.com/NetManAIOps/KPI-Anomaly-Detection)  
TraceAnomaly Datasets:[https://github.com/NetManAIOps/TraceAnomaly](https://github.com/NetManAIOps/TraceAnomaly)  
OmniAnomaly Datasets:[https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)     
PreFix DataSets:[https://github.com/NetManAIOps/PreFix](https://github.com/NetManAIOps/PreFix)    
Bagel Datasets:[https://github.com/NetManAIOps/Bagel](https://github.com/NetManAIOps/Bagel)   
Squeeze Datasets:[https://github.com/NetManAIOps/Squeeze](https://github.com/NetManAIOps/Squeeze)     
Dount Datasets:[https://github.com/NetManAIOps/donut](https://github.com/NetManAIOps/donut) 
##<span id="14">Open Source Code</span>
[Java] [Surus](https://github.com/Netflix/Surus):A collection of tools for analysis in Pig and Hive.  
[Python] [TODS](https://github.com/datamllab/tods): TODS is a full-stack automated machine learning system for outlier detection on multivariate time-series data.  
[Python] [skyline](https://github.com/earthgecko/skyline): Skyline is a near real time anomaly detection system.  
[Python] [banpei](https://github.com/tsurubee/banpei): Banpei is a Python package of the anomaly detection.  
[Python] [telemanom](https://github.com/khundman/telemanom): A framework for using LSTMs to detect anomalies in multivariate time series data.  
[Python] [DeepADoTS](https://github.com/KDD-OpenSource/DeepADoTS): A benchmarking pipeline for anomaly detection on time series data for multiple state-of-the-art deep learning methods.  
[Python] [NAB](https://github.com/numenta/NAB): The Numenta Anomaly Benchmark: NAB is a novel benchmark for evaluating algorithms for anomaly detection in streaming, real-time applications.  
[R] [AnomalyDetection](https://github.com/twitter/AnomalyDetection): AnomalyDetection is an open-source R package to detect anomalies which is robust, from a statistical standpoint, in the presence of seasonality and an underlying trend.  
[R] [anomalize](https://cran.r-project.org/web/packages/anomalize/): The 'anomalize' package enables a "tidy" workflow for detecting anomalies in data. 
###<span id="15">Credits</span>
- Anomaly detection [Resources](https://github.com/yzhao062/anomaly-detection-resources)
- Log survey [Logpai](https://github.com/logpai/log-survey)
##<span id="16">License</span>     





# Papers
Collection of papers in trajectory forecasting categorised according to the high-level structure

![version](https://img.shields.io/badge/version-0.0.1-ff69b4.svg) ![LastUpdated](https://img.shields.io/badge/LastUpdated-2020.04.15-lightgrey.svg) ![topic](https://img.shields.io/badge/topic-trajectory--prediction-brightgreen.svg?logo=github) [![HitCount](http://hits.dwyl.com/theDebugger811/trajectory-prediction-papers.svg)](http://hits.dwyl.com/theDebugger811/trajectory-prediction-papers)

The literature survey is categorized as:
1. Classical: Papers not utilizing neural networks for trajectory forecasting
2. Motion-Based: Papers utilizing neural networks for trajectory forecasting without modelling interactions with neighbouring agents or physical spaces.
3. Agent-Agent Interactions: Papers utilizing neural networks for trajectory forecasting modelling interactions with neighbouring agents but not physical spaces. 
4. Agent-Space Interactions: Papers utilizing neural networks for trajectory forecasting modelling interactions with physical spaces but not neighbouring agents.
5. Agent-Agent-Space Interactions: Papers utilizing neural networks for trajectory forecasting modelling interactions with both physical spaces as well as neighbouring agents.
6. Miscellaneous: Papers related to related topics like activity forecasting, human body dynamics

## Classical 
1. Social Force Model for Pedestrian Dynamics, 1998 [Paper](https://arxiv.org/pdf/cond-mat/9805244.pdf)
2. Simulation of pedestrian dynamics using a two-dimensional cellular automaton, 2001 [Paper](https://arxiv.org/pdf/cond-mat/0102397.pdf)
3. Discrete Choice Models for Pedestrian Walking Behavior, 2006 [Paper](https://infoscience.epfl.ch/record/77526/files/Antonini2004_721.pdf)
4. Continuum crowds, 2006 [Paper](https://grail.cs.washington.edu/projects/crowd-flows/78-treuille.pdf)
5. Modelling Smooth Paths Using Gaussian Processes, 2007 [Paper](https://hal.inria.fr/inria-00181664/file/Paper.pdf)
6. Reciprocal n-body Collision Avoidance (ORCA), 2008 [Paper](http://gamma.cs.unc.edu/ORCA/publications/ORCA.pdf)
7. You’ll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking, 2009 [Paper](http://vision.cse.psu.edu/courses/Tracking/vlpr12/PellegriniNeverWalkAlone.pdf)
8. Socially-Aware Large-Scale Crowd Forecasting, 2014 [Paper](http://vision.stanford.edu/pdf/alahi14.pdf)
9. Learning to Predict Trajectories of Cooperatively Navigating Agents, 2014 [Paper](http://www2.informatik.uni-freiburg.de/~kretzsch/pdf/kretzschmar14icra.pdf)
10. Understanding pedestrian behaviors from stationary crowd groups, 2015 [Paper](https://www.zpascal.net/cvpr2015/Yi_Understanding_Pedestrian_Behaviors_2015_CVPR_paper.pdf)
11. Learning Social Etiquette: Human Trajectory Understanding In Crowded Scenes, 2016 [Paper](https://infoscience.epfl.ch/record/230262/files/ECCV16social.pdf)
12. Point-based Path Prediction from Polar Histograms, 2016 [Paper](https://www.semanticscholar.org/paper/Point-based-path-prediction-from-polar-histograms-Coscia-Castaldo/37f35a05733e11cd490897a3c6d906abfe5ce434)

## Motion-Based 

1. Bi-Prediction: Pedestrian Trajectory Prediction Based on Bidirectional LSTM Classification, 2017 [Paper](https://www.researchgate.net/publication/322001876_Bi-Prediction_Pedestrian_Trajectory_Prediction_Based_on_Bidirectional_LSTM_Classification)
2. RED: A simple but effective Baseline Predictor for the TrajNet Benchmark, 2018 [Paper](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11131/Becker_RED_A_simple_but_effective_Baseline_Predictor_for_the_TrajNet_ECCVW_2018_paper.pdf)
3. Convolutional Neural Network for Trajectory Prediction, 2018 [Paper](https://arxiv.org/pdf/1809.00696.pdf)
3. Location-Velocity Attention for Pedestrian Trajectory Prediction, 2019 [Paper](https://www.researchgate.net/publication/331607165_Location-Velocity_Attention_for_Pedestrian_Trajectory_Prediction)
4. The Simpler the Better: Constant Velocity for Pedestrian Motion Prediction, 2019 [Paper](https://www.researchgate.net/publication/331887977_The_Simpler_the_Better_Constant_Velocity_for_Pedestrian_Motion_Prediction)
4. Transformer Networks for Trajectory Forecasting, 2020 [Paper](https://arxiv.org/pdf/2003.08111.pdf)

## Agent-Agent Interaction 
1. Social LSTM: Human Trajectory Prediction in Crowded Spaces, 2016 [Paper](https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf)
2. A Data-driven Model for Interaction-Aware Pedestrian Motion Prediction in Object Cluttered Environments, 2017 [Paper](https://arxiv.org/abs/1709.08528)
3. Soft + Hardwired Attention: An LSTM Framework for Human Trajectory Prediction and Abnormal Event Detection, 2017 [Paper](https://arxiv.org/pdf/1702.05552.pdf)
3. Social Attention: Modeling Attention in Human Crowds, 2017 [Paper](https://arxiv.org/abs/1710.04689) 
4. 3DOF Pedestrian Trajectory Prediction Learned from Long-Term Autonomous Mobile Robot Deployment Data, 2017 [Paper](http://iliad-project.eu/wp-content/uploads/2018/03/Kevin_UoL_ICRA18.pdf)
4. Encoding Crowd Interaction with Deep Neural Network for Pedestrian Trajectory Prediction, 2018 [Paper](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2136.pdf)
4. Group LSTM: Group Trajectory Prediction in Crowded Scenarios, 2018 [Paper](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11131/Bisagno_Group_LSTM_Group_Trajectory_Prediction_in_Crowded_Scenarios_ECCVW_2018_paper.pdf)
5. MX-LSTM: mixing tracklets and vislets to jointly forecast trajectories and head poses, 2018 [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hasan_MX-LSTM_Mixing_Tracklets_CVPR_2018_paper.pdf)
5. StarNet: Pedestrian Trajectory Prediction using Deep Neural Network in Star Topology, 2019 [Paper](https://arxiv.org/abs/1906.01797)
6. SR-LSTM: State Refinement for LSTM towards Pedestrian Trajectory Prediction, 2019 [Paper](https://arxiv.org/abs/1903.02793)
7. Recursive Social Behavior Graph for Trajectory Prediction, 2020 [Paper](https://arxiv.org/pdf/2004.10402.pdf)
9. Collaborative Motion Prediction via Neural Motion Message Passing [Paper](https://arxiv.org/pdf/2003.06594.pdf)

### Multimodal

1. Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks, 2018 [Paper](https://arxiv.org/pdf/1803.10892.pdf)
2. Social Ways: Learning Multi-Modal Distributions of Pedestrian Trajectories with GANs, 2019 [Paper](https://arxiv.org/pdf/1904.09507.pdf)
3. Which Way Are You Going? Imitative Decision Learning for Path Forecasting in Dynamic Scenes, 2019 [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Which_Way_Are_You_Going_Imitative_Decision_Learning_for_Path_CVPR_2019_paper.pdf)
4. Analyzing the Variety Loss in the Context of Probabilistic Trajectory Prediction, 2019 [Paper](https://arxiv.org/pdf/1907.10178.pdf)
5. The Trajectron: Probabilistic Multi-Agent Trajectory Modeling With Dynamic Spatiotemporal Graphs, 2019 [Paper](https://arxiv.org/abs/1810.05993) 
6. STGAT: Modeling Spatial-Temporal Interactions for Human Trajectory Prediction, 2019 [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_STGAT_Modeling_Spatial-Temporal_Interactions_for_Human_Trajectory_Prediction_ICCV_2019_paper.pdf)
7. Stochastic Trajectory Prediction with Social Graph Network, 2019 [Paper](https://arxiv.org/pdf/1907.10233.pdf)
8. Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural
Network for Human Trajectory Prediction, 2020 [Paper](https://arxiv.org/pdf/2002.11927.pdf)
9. It Is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction, 2020 [Paper](https://arxiv.org/pdf/2004.02025.pdf)
10. STAR: Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction, 2020 [Paper](https://arxiv.org/pdf/2005.08514.pdf)

## Agent-Agent-Space Interaction 

1. Context-Aware Trajectory Prediction in Crowded Spaces, 2017 [Paper](https://arxiv.org/pdf/1705.02503.pdf)
2. Human Trajectory Prediction using Spatially aware Deep Attention Models, 2017 [Paper](https://arxiv.org/pdf/1705.09436.pdf)
2. SS-LSTM: A Hierarchical LSTM Model for Pedestrian Trajectory Prediction, 2018 [Paper](https://ieeexplore.ieee.org/document/8354239)
3. A Data-driven Model for Interaction-aware Pedestrian Motion Prediction in Object Cluttered Environments, 2018 [Paper](https://arxiv.org/pdf/1709.08528.pdf)
3. Multi-Agent Tensor Fusion for Contextual Trajectory Prediction, 2019 [Paper](https://arxiv.org/pdf/1904.04776.pdf)

### Multimodal
1. DESIRE: Distant Future Prediction in Dynamic Scenes with Interacting Agents, 2017 [Paper](https://arxiv.org/pdf/1704.04394.pdf)
2. SoPhie: An Attentive GAN for Predicting Paths Compliant to Social and Physical Constraints, 2019 [Paper](https://arxiv.org/pdf/1806.01482.pdf)
3. Peeking into the Future: Predicting Future Person Activities and Locations in Videos, 2019 [Paper](https://arxiv.org/pdf/1902.03748.pdf)
4. Social-BiGAT: Multimodal Trajectory Forecasting using Bicycle-GAN and Graph Attention Networks, 2019 [Paper](https://arxiv.org/abs/1907.03395)
4. Social-WaGDAT: Interaction-aware Trajectory Prediction via Wasserstein Graph Double-Attention Network, 2020 [Paper](https://arxiv.org/pdf/2002.06241.pdf)
5. Trajectron++: Multi-Agent Generative Trajectory Forecasting With Heterogeneous Data for Control, 2020 [Paper](https://arxiv.org/abs/1810.05993)
6. Reciprocal Learning Networks for Human Trajectory Prediction, 2020 [Paper](https://arxiv.org/pdf/2004.04340.pdf)
7. The Garden of Forking Paths: Towards Multi-Future Trajectory Prediction, 2020 [Paper](https://arxiv.org/pdf/1912.06445.pdf)
8. Dynamic and Static Context-aware LSTM for Multi-agent Motion Prediction, 2020 [Paper](https://arxiv.org/pdf/2008.00777.pdf)

## Agent-Space Interaction 
### Multimodal
1. Accurate and Diverse Sampling of Sequences based on a “Best of Many” Sample Objective, 2018 [Paper](https://arxiv.org/pdf/1806.07772.pdf)
2. Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction, 2019 [Paper](https://arxiv.org/pdf/1906.03631.pdf)
3. Scene Compliant Trajectory Forecast with Agent-Centric Spatio-Temporal Grids, 2020 [Paper](https://www.semanticscholar.org/paper/Scene-Compliant-Trajectory-Forecast-With-Grids-Ridel-Deo/e2bfb1b90000e19b4bca6a7f8aab5f6305c6a2be)
4. Goal-GAN: Multimodal Trajectory Prediction Based on Goal Position Estimation, 2020 [Paper](https://arxiv.org/pdf/2010.01114.pdf)

## Miscellaneous

### Activity Forecasting
1. Trajectory Learning for Activity Understanding: Unsupervised, Multilevel, and Long-Term Adaptive Approach, 2011 [Paper](https://www.researchgate.net/publication/50596076_Trajectory_Learning_for_Activity_Understanding_Unsupervised_Multilevel_and_Long-Term_Adaptive_Approach)
1. Activity forecasting, 2012 [Paper](https://www.ri.cmu.edu/pub_files/2012/10/Kitani-ECCV2012.pdf) 
2. Context-Based Pedestrian Path Prediction, 2014 [Paper](http://www.gavrila.net/eccv14.pdf) 
3. Pedestrian’s Trajectory Forecast in Public Traffic with Artificial Neural Networks, 2014 [Paper](https://www.researchgate.net/publication/269635918_Pedestrian's_Trajectory_Forecast_in_Public_Traffic_with_Artificial_Neural_Networks)
2. Learning Intentions for Improved Human Motion Prediction, 2014 [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0921889014000062)
5. Pedestrian Path, Pose, and Intention Prediction Through Gaussian Process Dynamical Models and Pedestrian Activity Recognition, 2019 [Paper](https://www.researchgate.net/publication/325495601_Pedestrian_Path_Pose_and_Intention_Prediction_Through_Gaussian_Process_Dynamical_Models_and_Pedestrian_Activity_Recognition)

### Human Body Dynamics
1. Gaussian Process Dynamical Models for Human Motion, 2008 [Paper](http://www.dgp.toronto.edu/~jmwang/gpdm/pami_with_errata.pdf)
1. Recurrent Network Models for Human Dynamics, 2015 [Paper](https://arxiv.org/abs/1508.00271)

# Evaluation

Comparison of popular human trajectory forecasting papers based on the datasets on which the methods have been evaluated. 

| Method            | ETH/UCY |    SDD  | TrajNet++ | Multipath |
|:-----------------:|:-------:|:-------:|:---------:|:---------:|
| S-LSTM            | &check; |         |           |           |
| DESIRE            |         | &check; |           |           |
| S-GAN             | &check; |         |           |           |
| Sophie            | &check; | &check; |           |           |
| Trajectron        | &check; |         |           |           |
| Social-BiGAT      | &check; |         |           |           |
| Social-STGCNN     | &check; |         |           |           |
| Multiverse        |         |         |           |  &check;  |
| PECNet            | &check; | &check; |  &check;  |           | 
| D-LSTM            |         |         |  &check;  |           |
| Social-NCE        |         |         |  &check;  |           |

### A Note on Evaluation benchmarks
Evaluation on TrajNet++ is preferred in comparison to ETH/UCY as the test set and the evaluation protocol for TrajNet++ is fixed (and extensive!). More details [here](https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge). The variation in ADE/FDE greatly reduces among different methods when evaluated on equal grounds on TrajNet++ ([leaderboard](https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge/leaderboards)) in comparison to the numbers reported on ETH/UCY.

# Trajectory Forecasting Framework
If you are new to trajectory forecasting, do check out the [TrajNet++](https://github.com/vita-epfl/trajnetplusplusbaselines) framework! TrajNet++ is a code-base with specific focus on human trajectory forecasting, and having more than 10 trajectory forecastng baselines already implemented. 



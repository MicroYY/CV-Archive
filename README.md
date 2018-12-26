# CV-Archive
My personal paper/code/dataset archives about computer vision

## 目录
* [Paper](#Paper)
* [Code](#Code)
* [Dataset](#Dataset)

### Paper

#### 室外语义重建
|#|Paper|Published in|Summary|Misc|
|-|----|----|---|---|
|1|[Learning Priors for Semantic 3D Reconstruction](http://cvg.ethz.ch/research/learned-regularization/)|ECCV 2018||也适用于室内;暂时未开源|
|2|[RayNet: Learning Volumetric 3D Reconstruction with Ray Potentials](https://avg.is.tue.mpg.de/research_projects/raynet)|CVPR 2018|基于multi-view的考虑，在learning-based重建的基础上加入了几何约束|非语义重建；[已开源](https://github.com/paschalidoud/raynet)|
|3|[Robust Dense Mapping for Large-Scale Dynamic Environments](https://siegedog.com/dynslam/)|ICRA 2018|SLAM + semantic segmentation + scene flow|双目;[已开源](https://github.com/AndreiBarsan/DynSLAM)|
|4|[Semantic Multi-view Stereo: Jointly Estimating Objects and Voxels](http://ps.is.tue.mpg.de/project/Volumetric_Reconstruction)|CVPR 2017|引入shape prior解决遮挡，进行probabilistic重建|
|5|Semantically Informed Multiview Surface Refinement|ICCV 2017|3D网格几何表面和语义标签的联合优化|[已开源](https://bitbucket.org/mathiaro/meshref/)|
|6|[Semantic 3D reconstruction with continuous regularization and ray potentials using a visibility consistency constraint](https://www.nsavinov.com/publication/2016-continuous-ray/)|CVPR 2016|
|7|[Discrete Optimization of Ray Potentials for Semantic 3D Reconstruction](https://www.nsavinov.com/publication/2015-discrete-ray/)|CVPR 2015|直接优化三维模型的投影和图像的误差||
|8|[Joint 3D Scene Reconstruction and Class Segmentation](https://people.eecs.berkeley.edu/~chaene/publications.html)|CVPR 2013|提出了一个分割和重建联合优化解决的框架|
|9|An overview of recent progress in volumetric semantic 3D reconstruction|ICPR 2016|语义三维重建的综述||
|10|Dense Semantic 3D Reconstruction|TPAMI 2017|同8||
|11|Large-Scale Outdoor 3D Reconstruction on a Mobile Device|CVIU 2017|移动设备（Google Project Tango)上的大规模重建|基本real-time|
|12|Incremental Dense Semantic Stereo Fusion for Large-Scale Semantic Scene Reconstruction|ICRA 2015|第一个实时的大范围室外场景稠密语义重建|
|13|Large-Scale Semantic 3D Reconstruction: An Adaptive Multi-Resolution Model|CVPR 2016|提出了一种自适应的multi-resolution的框架|离线处理|
|14|[Joint Semantic Segmentation and 3D Reconstruction from Monocular Video](http://abhijitkundu.info/projects/JointSegRec/index.html)|ECCV 2014|构建体素块的离散图重建街景|单目，不需要估计深度|



#### 室内语义重建
|#|Paper|Published in|Summary|Misc|
|---|----|-----|-----|-|
|1|[3D Semantic Parsing of Large-Scale Indoor Spaces](http://buildingparser.stanford.edu/index.html)|CVPR 2016|大规模室内场景3D语义解析|提供了数据集|

#### Object语义重建
|#|Paper|Published in|Summary|Misc|
|-|-|-|-|-|
|1|Dense Object Reconstruction with Semantic Priors|CVPR 2013|通过目标检测和形状先验引入语义信息，克服了传统MVS的缺点|又多视角图片训练出物体的语义先验|
|2|[Segment Based 3D Object Shape Priors](http://cvg.ethz.ch/research/semantic-3d-modeling/)|CVPR 2015|提出了新的shape prior formulation，将物体分成多个凸块,重建问题可以看作容积式的多标签分割||
|3|[Class Specific 3D Object Shape Priors Using Surface Normal](https://www.nsavinov.com/publication/2014-shape-priors/)|CVPR 2014|利用物体表面的法向信息作为shape prior|
|4|[Learning a Multi-view Stereo Machine](https://bair.berkeley.edu/blog/2017/09/05/unified-3d/)|NIPS 2017|一个端到端的网络，输入多视角图片，输出Voxel Occupancy Grid或者深度图|[已开源](https://github.com/akar43/lsm)|

#### Stereo
|#|Paper|Published in|Summary|Misc|
|-|-|-|-|-|
|7|Direction Matters: Depth Estimation with a Surface Normal Classifier|CVPR 2015|在depth estimation中引入了表面法向量分类||







### Dataset

#### 3D数据集
|Dataset|Hyperlink|Comment|
|-|-|-|
|3D Warehouse|https://3dwarehouse.sketchup.com/|有一些建筑模型|

#### 2D数据集
##### Semantic segmentation
|Dataset|Hyperlink|Paper|Comment|
|-|-|-|-|
|NYU Depth Dataset V2|https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html||RGB + D + class labels|
|Semantic3D|http://www.semantic3d.net/|Semantic3D: A new Large-scale Point Cloud Classification Benchmark|大规模点云分类的benchmark|

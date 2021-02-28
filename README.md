# Robust-Depth-Completion-in-Time-Seasonal-Changing-Condition
  * The Adaptive Instance Normalization (AdaIN) used for Style Transfer has been applied to the existing depth completion model so that depth information can be strongly estimated in a rain-night environment. 
  * As a result, we found that the accuracy of depth completion increased from about 77% to 80%. 

# Motivation
* Studies currently underway have set clear indoor or day outdoor images as datasets and conducted experiments.
* It is difficult to estimate depth information of image in Rain-Night environment.

# Model
![models](https://user-images.githubusercontent.com/78299113/109423781-2601de00-7a24-11eb-9402-f60065144113.png)
**Figure 1.** Our network model. 4 layers of Adaptive Instance Normalization (AdaIN) are inserted in baseline model.

# Result
### Network Evaluation

**Table 1.** The results comparison on the RAINNIGHT dataset according to the number of Adaptive Instance Normalization (AdaIN) layer and whether a parameter α in Adaptive Instance Normalization (AdaIN) part was set to learnable parameter. Baseline is the model of [1]. Ours #1 and Ours #2 are our model.
![table 1](https://user-images.githubusercontent.com/78299113/109424461-0a4c0700-7a27-11eb-9b50-8d9b87223338.png)

![figure 2](https://user-images.githubusercontent.com/78299113/109424493-2b145c80-7a27-11eb-8b8e-2b09a42f361f.png)
**Figure 2.** Results for Ours #2. From the left are RGB image, sparse depth sample, ground truth, model prediction. The above is the result of the test with Virtual KITTI 2 daytime datasets, and the below is the result of the test with SYNTHIA RAINNIGHT datasets.

### Comparison with the Baseline

**Table 2.** The comparison of the baseline [1] model and our model on the three image datasets when testing. Performance was similarly high for the Day dataset, and performance of our model was slightly higher for Winter and Rain Night datasets.
![table 2](https://user-images.githubusercontent.com/78299113/109424611-a118c380-7a27-11eb-9b06-b3df0c58af8c.png)

![figure 3](https://user-images.githubusercontent.com/78299113/109424623-ab3ac200-7a27-11eb-8251-8c07434bcbf6.png)
**Figure 3.** The comparison of the test results of baseline [1] model (above) and our model (below) on the RAINNIGHT datasets. It can be seen that the image below predicted the depth information better in the lower right part.

### Result video
https://www.youtube.com/watch?v=GUoZR_Q06Vg&feature=youtu.be

# Reference
[1]  Ma, Fangchang and S. Karaman. “Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image.” 2018 IEEE International Conference on Robotics and Automation (ICRA) (2018): 1-8.

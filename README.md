# Robust-Depth-Completion-in-Time-Seasonal-Changing-Condition 
  * Style Transfer에 사용되는 Adaptive Instance Normalization (AdaIN) 을 기존 깊이 추정 모델에 적용하여 비 오는 밤 환경에서도 깊이 정보가 강인하게 추정되도록 했습니다.
  * 결과적으로, 깊이 정보 추정 정확도를 77%에서 80%로 높였습니다.



## Motivation

* 현재 진행되고 있는 연구들은 밝은 실내 이미지 또는 낮 야외 이미지를 대상으로 합니다.
* 비 오는 밤 환경에서 이미지의 깊이 정보를 추정하는 일은 어렵습니다.



## Model

![models](https://user-images.githubusercontent.com/78299113/109423781-2601de00-7a24-11eb-9402-f60065144113.png)
**Figure 1.** 개선된 네트워크 모델. Adaptive Instance Normalization (AdaIN) layer 4개가 baseline model에 삽입된 구조.



## Result

### Network Evaluation

**Table 1.** 다음 표는 Adaptive Instance Normalization (AdaIN) layer의 개수와 parameter α가 learnable parameter로 설정되었는지의 여부에 따른 결과입니다. Baseline model은 논문 [1]의  model입니다. Ours #1 와 Ours #2는 이번 프로젝트를 통해 만들어진 모델입니다.

|          | Num. of AdaIN | Parameter α |    RMSE    |    REL    |    MAE     | δ <sub>1</sub> |
| -------- | :-----------: | :---------: | :--------: | :-------: | :--------: | :------------: |
| Baseline |       -       |      -      |   92.632   | **0.328** |   31.467   |     0.735      |
| Ours #1  |       4       |      1      | **71.222** |   0.371   | **23.661** |     0.735      |
| Ours #2  |       4       |  Learnable  |   71.932   |   0.381   |   24.129   |   **0.744**    |

![figure 2](https://user-images.githubusercontent.com/78299113/109424493-2b145c80-7a27-11eb-8b8e-2b09a42f361f.png)
**Figure 2.** Ours #2의 결과입니다. 왼쪽부터 RGB image, sparse depth sample, ground truth, model prediction입니다. 위쪽은 Virtual KITTI 2 daytime datasets으로 실험한 결과이고 아래쪽은 SYNTHIA RAINNIGHT datasets으로 실험한 결과입니다. 



### Comparison with the Baseline

**Table 2.** 세 종류의 이미지 데이터셋에 대한 baseline model [1]과 개선된 모델의 실험 결과입니다. Day dataset에 대해서는 두 모델의 성능이 비슷한 정도로 높지만 RAINNIGHT dataset과 WINTER dataset에 대해서는 개선된 모델의 성능이 더 높습니다.

|                | Day (Virtual KITTI 2) | Day (Virtual KITTI 2) | Winter (SYNTHIA) | Winter (SYNTHIA) | Rain Night (SYNTHIA) | Rain Night (SYNTHIA) |
| :------------: | :-------------------: | :-------------------: | :--------------: | :--------------: | :------------------: | :------------------: |
|                |       Baseline        |         Ours          |     Baseline     |       Ours       |       Baseline       |         Ours         |
|      RMSE      |        11.790         |        11.443         |      15.431      |    **14.720**    |        15.358        |      **14.673**      |
|      REL       |         0.069         |         0.074         |      0.256       |      0.210       |        0.241         |        0.208         |
|      MAE       |         3.072         |         3.116         |      6.715       |      5.775       |        6.579         |        5.767         |
| δ <sub>1</sub> |       **0.948**       |       **0.947**       |      0.771       |    **0.809**     |        0.771         |      **0.806**       |

![figure 3](https://user-images.githubusercontent.com/78299113/109424623-ab3ac200-7a27-11eb-8251-8c07434bcbf6.png)
**Figure 3.** RAINNIGHT datasets에 대한 baseline model [1] (위쪽)과 개선된 모델 (아래쪽)의 실험 결과입니다. 이미지의 오른쪽 아랫 부분을 보면 개선된 모델 (아래쪽)의 실험 결과가 깊이 정보를 더 잘 예측한 것을 볼 수 있습니다.



### Result video
링크를 통해 결과 영상을 볼 수 있습니다.
https://www.youtube.com/watch?v=GUoZR_Q06Vg&feature=youtu.be



## Reference

[1]  Ma, Fangchang and S. Karaman. “Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image.” 2018 IEEE International Conference on Robotics and Automation (ICRA) (2018): 1-8.
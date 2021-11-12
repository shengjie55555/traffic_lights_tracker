# Traffic Lights Tracker
## 1 系统组成
整个系统包括三个模块：检测、跟踪、滤波和匹配。
### 1.1 检测与跟踪
参考该仓库[YOLOv5+DeepSORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch).主要改动在于增加了track的属性:
```python
self.box = box
self.score = score
self.cls = []
if cls is not None:
    self.cls.append(cls)
```
其中将cls变成列表，用于保存多帧的检测结果，取最新5帧的检测结果中的众数作为该目标id检测结果，这是第一次滤波。
### 1.2 滤波与匹配
根据规划模块发送的交通灯个数num，将检测结果和实际感兴趣的交通灯进行匹配。  
step1：先按照左上顶点的y坐标**从小到大**（保证搜索到的第一个目标为感兴趣的交通灯）排序，然后从上到下搜索，直到找到boxes，计算找到的boxes平均的y坐标:y0，再次搜索[0, y0+10]区域内的boxes  
step2：如果个数大于num，选择概率最大的num个，用相对位置从左到右匹配  
step3：如果个数小与num，首先根据id匹配之前的结果，新出现的目标则先**从左到右**（初始化后全为0，此时计算的最佳匹配结果都是第0个）排序后，按照和之前结果的距离匹配  
匹配完成后，保存最新的5帧结果，取众数进行第二次滤波
## 2 代码解释
### 2.1 get_camera
```shell
camera_publisher: 加载本地视频文件，发送/camera/rgb/image_pointgrey
camera_subscriber: 订阅/camera/rgb/image_pointgrey
ros_driver_pointgrey_camera: 工业相机的ros驱动  
ros_driver_gmsl2_camera: gmsl2相机的ros驱动
video: 采集工业相机的数据，保存成本地视频
```
备注：由于ros版本默认采用python2，因此无法使用cv_bridge
### 2.2 tracker
```shell
pointgrey_visual_detection: 订阅/camera/rgb/image_pointgrey,再进行检测
gmsl2_visual_detection: 订阅/camera/rgb/image_gmsl2,再进行检测
```
使用时需要通过Traffic_Light_Pos_Pub.cpp发送交通灯个数，数量为argv[0]
```shell
rosrun tracker tracker 2  # num = 2
```  
## 3 采集数据
方法1：
```shell
rosrun get_camera video.py  # 直接打开驱动，保存视频
```
方法2：
```shell
rosrun get_camera ros_driver_camera.py  # 打开驱动，发布话题
rosrun get_camera camera_subscriber.py  # 订阅话题，如果要保存视频，在该文件内搜索todo修改路径
```
## 4 使用步骤
步骤1：下载仓库，编译
```shell
cd Traffic_Lights_Tracker
catkin_make
source ./devel/setup.bash
```
步骤2：拷贝权重
```shell
cp /path_to_yolo_weights/best.pt ./src/tracker/scripts/
cp /path_to_deep_sort_weights/ckpt.t7 ./src/tracker/scripts/
```
步骤3：修改路径
```shell
打开./src/tracker/scripts/common.py，搜索todo，总共有两次，修改实际权重和配置文件的路径
```
步骤4：运行
```shell
roslaunch tracker gmsl2_visual_detection.launch  # 打开驱动和检测程序
rosrun tracker tracker 3  # 用于发送lights数量，0：关闭检测
```
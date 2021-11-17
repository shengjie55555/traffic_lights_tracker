# Traffic Lights Tracker
## 1 代码解释
### 1.1 get_camera
```shell
camera_publisher: 加载本地视频文件，发送/camera/rgb/image_pointgrey
camera_subscriber: 订阅/camera/rgb/image_pointgrey
ros_driver_pointgrey_camera: 工业相机的ros驱动  
ros_driver_gmsl2_camera: gmsl2相机的ros驱动
video: 采集工业相机的数据，保存成本地视频
```
备注：由于ros版本默认采用python2，因此无法使用cv_bridge
### 1.2 detector
```shell
detector: 订阅/camera/rgb/image_pointgrey,再进行检测
```
## 2 使用步骤
步骤1：下载仓库，编译
```shell
cd Traffic_Lights_Tracker
catkin_make
source ./devel/setup.bash
```
步骤2：拷贝权重
```shell
cp /path_to_yolo_weights/best.pt ./src/detector/scripts/
```
步骤3：修改路径
```shell
打开./src/detector/scripts/detector.py  # 搜索todo，修改实际权重的路径
```
步骤4：运行
```shell
rosrun get_camera ros_driver_pointgrey_camera  # 打开相机驱动
rosrun detector detector  # 检测
```
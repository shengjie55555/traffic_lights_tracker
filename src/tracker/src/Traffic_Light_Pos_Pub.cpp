//
// Created by sheng on 2021/7/25.
//

#include <ros/ros.h>
#include <tracker/traffic_lights_num.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "traffic_light_num_pub");
    ros::NodeHandle nh;

    tracker::traffic_lights_num data;
    for(int i = 0; i < argc; i++)
    {
        ROS_INFO("argc: %d, argv: %s", i, argv[i]);
    }
    data.num = int(*argv[1] - '0');

    ros::Publisher pub = nh.advertise<tracker::traffic_lights_num>("Traffic_Lights_Num", 1);
    ros::Rate loop_rate(20);
    while(ros::ok())
    {
        ROS_INFO("num = %d", data.num);
        pub.publish(data);
        loop_rate.sleep();
    }
    return 0;
}

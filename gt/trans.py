#!/usr/bin/python
# -*- coding: UTF-8 -*-
import csv
# 输入csv文件名称和输出txt文件名称
#csv_file = '/home/zty/workspace/catkin_ws_ov/src/open_vins/gt/mh01.csv '
txt_file = '/home/zty/workspace/catkin_ws_ov/src/open_vins/gt/mh01.txt '
with open(txt_file, "w") as my_output_file:
    with open('/home/zty/workspace/catkin_ws_ov/src/open_vins/gt/mh01.csv', "r") as my_input_file:
        #逐行读取csv存入txt中
        for row in csv.reader(my_input_file):
            # 前8个数据是:timestamp tx ty tz qw qx qy qz
            row = row[0:8]
            # 时间戳单位处理
            temp1 = row[0][0:10] + '.' + row[0][10:16]
            row[0] = temp1
            # 互换 qw 和 qx
            temp2 = row[4]
            row[4] = row[7]
            row[7] = temp2
            my_output_file.write(" ".join(row)+'\n')
    my_output_file.close()

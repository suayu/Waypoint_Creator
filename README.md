# Waypoint_Creator

此项目基于manual_control.py脚本实现，增加了车辆轨迹记录和回放功能。

# Usage

使用`manual_control.py`替代`Carla/PythonAPI/examples/manual_control.py`这一Carla自带的原生同名文件。

将`get_tf.py`放入`Carla/PythonAPI/examples/`目录。

启动脚本：<br>
`python manual_control.py --save_path=$PATH`

如：<br>
`python manual_control.py --save_path=tf.npy`

脚本启动后，在窗口内按下`E`开始记录车辆Waypoint，再次按下`E`结束记录并把记录以npy格式保存至`$PATH`文件中。

读取Waypoint：<br>
`python get_tf.py --read_path=$PATH --dT=$dT`

脚本启动后，将会读取`$PATH`文件记录的npy格式的自车Waypoint数据，以`dT`为读取的时间间隔，插值输出自车的Transform信息。

或者，可以在你的代码中调用get_tf.py的方法：<br>
```bash
import get_tf

transform, time_stamp = get_tf.read_data($PATH)
x, y, z, pitch ,yaw, roll = get_tf.interpolate(transform, time_stamp, 0.1)
```

其中，`x, y, z, pitch ,yaw, roll`均为shape为`(n,)`的ndarray格式数组，每个元素保存一个时间间隔开始时自车的Transform属性值。

保存的中间npy格式数据格式为：<br>
`shape`:(帧数,10)<br>
每个元素：<br>
`(x、y、z、pitch、yaw、roll、frame、elapsed_seconds、delta_seconds、platform_timestamp)`<br>
`frame`：模拟器启动以来经过帧数<br>
`elapsed_seconds`：仿真经过秒数<br>
`delta_seconds`：从上一帧开始经过秒数<br>
`platform_timestamp`：以秒为单位给出测量帧的寄存器<br>
上一帧的`elapsed_seconds`+这一帧的`delta_seconds`=这一帧的`elapsed_seconds`
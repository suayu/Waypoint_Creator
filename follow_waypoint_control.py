import math
import time
import random
import carla
import os
import scipy
import get_tf
import numpy as np
import runtime
import PurePursuit

# 参考点最大间距
THRESHOLD = 1.5
# 参考点期望间距
TARGET_DIS = 0.2

def calculate_yaws(x, y):
    yaws = []
    for i in range(len(x) - 1): 
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        yaw = math.atan2(dx, dy)
        yaw = math.degrees(yaw)
        if yaw > 90:
            yaw = - (yaw - 90)
        else:
            if -90 <= yaw <= 90:
                yaw = 90 - yaw
            else:
                yaw = - yaw - 270
        yaws.append(yaw)
    return yaws 

def main(data_path = "tf.npy", forward = True, create_ego_car = True, reference_points = []):
    dT = 0.02

    if not len(reference_points) == 0:

        # 有轨迹输入，采用轨迹输入
        temp_x = np.array([point[0] for point in reference_points])
        temp_y = np.array([point[1] for point in reference_points])
        z = np.array([0])
        # print(np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1))

        # 简单插值 不可用
        # # 轨迹总长度
        # path_length = 0
        # for i in range(0,len(temp_x)-1):
        #     path_length += math.sqrt((temp_x[i]-temp_x[i+1])*(temp_x[i]-temp_x[i+1])+(temp_y[i]-temp_y[i+1])*(temp_y[i]-temp_y[i+1]))

        # # 生成一批新的输入
        # x = np.linspace(np.amin(temp_x), np.amax(temp_x), int(path_length) * 10) # 从[0,10]这个范围等间隔取100个值
        # # 基于离散数组xp和fp的映射关系(正弦函数)，对新的输入数组x预测对应的输出y(线性插值)
        # y = np.interp(x, temp_x, temp_y)
        # new_len = len(x)

        # print("ref:",reference_points)

        # 考虑到参考路径点可能过于稀疏，进行上采样，以方便PurePursuit算法追踪
        path_length = 0
        x_ = []
        y_ = []
        for i in range(0,len(temp_x)-1):
            dis = math.sqrt((temp_x[i]-temp_x[i+1])*(temp_x[i]-temp_x[i+1])+(temp_y[i]-temp_y[i+1])*(temp_y[i]-temp_y[i+1]))
            path_length += dis
            # print("points:",[temp_x[i],temp_y[i]],[temp_x[i+1],temp_y[i+1]]," dis:",dis)
            if dis > THRESHOLD:
                target_dis = TARGET_DIS
                target_points_num = int(dis / target_dis) + 1

                xp = np.array([temp_x[i],temp_x[i+1]])
                yp = np.array([temp_y[i],temp_y[i+1]])

                xn = np.linspace(temp_x[i], temp_x[i+1], target_points_num)
                yn = np.interp(xn, xp, yp)

                x_.extend(xn.tolist())
                y_.extend(yn.tolist())
                # print("dis:",dis,"  ori:",[temp_x[i],temp_y[i]],[temp_x[i+1],temp_y[i+1]])
                # print("after:",xn.tolist(),yn.tolist())
                
            else:
                x_.append(temp_x[i])
                y_.append(temp_y[i])

            
            

        new_len = len(x_)
        x = np.array(x_)
        y = np.array(y_)
        # print("new:",x,"\n",y)
        yaw = calculate_yaws(x,y)

    elif len(reference_points) == 0 and os.path.exists(data_path):

        # 无轨迹输入，且文件存在，采用文件数据
        transform, time_stamp = get_tf.read_data(data_path)

        # 使用插值法修正轨迹
        x, y, z, pitch, yaw, roll = get_tf.interpolate(transform, time_stamp, dT)
        new_len = x.shape[0]

        # 轨迹总长度
        path_length = 0
        for i in range(0,new_len-1):
            path_length += math.sqrt((x[i]-x[i+1])*(x[i]-x[i+1])+(y[i]-y[i+1])*(y[i]-y[i+1]))
    else:

        # 无轨迹输入，且文件不存在，报错
        print("Failed to find reference path!")
        return

    

    # 使用PurePursuit算法修正轨迹
    path = PurePursuit.SimpleTest.get_path(x,y)
    # 前进把速度设成1，倒车把速度设成-1
    if forward == True:
        v = 1
    else:
        v = -1
    state = np.array([x[0], y[0], np.radians(yaw[0]), v]).astype(np.float64)
    test = PurePursuit.SimpleTest(initial_state = state)
    new_path = test.offline_test(path, math.floor(path_length/(v*runtime.dt))-1)
    # print("new_path:",new_path)
    x = [state[0] for state in new_path]
    y = [state[1] for state in new_path]
    yaw = [np.degrees(state[2]) for state in new_path]
    new_len = len(new_path)

    client = carla.Client('localhost', 2000)
    # client.load_world('SUSTech_COE_ParkingLot')
    # client.load_world('Town01')
    world = client.get_world()
    if create_ego_car == True:
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle_bp = world.get_blueprint_library().filter('*vehicle*').filter('vehicle.tesla.*')[0]
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    else:
        ego_vehicle = world.get_actors().filter('vehicle.*')[0]
    cur_location = carla.Location(x = x[0], y = y[0], z = z[0]-0.0035)

    world.get_spectator().set_transform(carla.Transform(cur_location+carla.Location(z=3),carla.Rotation(pitch=0)))

    for i in range(1,new_len):
        location = carla.Location(x = x[i], y = y[i], z = z[0]-0.0035)
        ego_vehicle.set_transform(carla.Transform(location,carla.Rotation(pitch=0,yaw=yaw[i],roll=0)))

        # velocity = carla.Vector3D(x = (x[i] - cur_location.x) / dT, y = (y[i] - cur_location.y) / dT, z = (z[i] - cur_location.z) / dT)
        velocity = carla.Vector3D(x = (x[i] - cur_location.x) / dT, y = (y[i] - cur_location.y) / dT, z = 0)
        v = math.sqrt(velocity.x * velocity.x + velocity.y*velocity.y + velocity.z*velocity.z)

        world.debug.draw_line(cur_location + carla.Location(z = runtime.H / 2), location + carla.Location(z = runtime.H / 2), thickness=0.1, color=carla.Color(r = 0, g = 0, b = 255), life_time = 10)
        cur_location = location
        time.sleep(dT)

if __name__ == "__main__":
    main()


import carla
import pygame
import time
import cv2
import math
import queue
import struct
import numpy as np
from PIL import Image

WORLD_NAME = 'Town01'

IMAGE_SIZE_X = 576
IMAGE_SIZE_Y = 300
LIDAR_HEIGHT = 2.4

RGB_FRONT = {
    "x" : 2,
    "y" : 0,
    "z" : 1.3,
    "yaw" : 0
}
RGB_REAR = {
    "x" : -2,
    "y" : 0,
    "z" : 1.3,
    "yaw" : 180
}
RGB_SIDE_FRONT = {
    "x" : 0,
    "y" : 2,
    "z" : 1.3,
    "yaw" : 45
}
RGB_SIDE_REAR = {
    "x" : 0,
    "y" : 2,
    "z" : 1.3,
    "yaw" : 120
}

queue_RGB_Front = queue.Queue()
queue_RGB_Rear = queue.Queue()
queue_RGB_Left_Front = queue.Queue()
queue_RGB_Right_Front = queue.Queue()
queue_RGB_Left_Rear = queue.Queue()
queue_RGB_Right_Rear = queue.Queue()
queue_LIDAR_TOP = queue.Queue()
queue_Semantic_LIDAR = queue.Queue()
queue_RADAR = queue.Queue()

# 将Carla的RGB相机传感器转化为ndarray格式
def convert_RGB(image):
    # img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape(
        (image.height, image.width, img.shape[0] // image.height // image.width)
    )
    img = img[:, :, :3]
    img = img[:, :, ::-1]

    return img

# 将Carla的LIDAR相机传感器转化为ndarray格式
def convert_LIDAR(data):
    points = np.frombuffer(data.raw_data, dtype=np.float32)
    points = np.reshape(points, (points.shape[0] // 4, 4))

    return points

# 将Carla的语义激光雷达真值转化为ndarray格式
def convert_Semantic_LIDAR(data):
    points = np.frombuffer(data.raw_data, dtype=np.float32)
    points = np.reshape(points, (points.shape[0] // 6, 6))
    label = np.frombuffer(points[:, 5].tobytes(), dtype=np.uint32)  # .astype('uint32')
    points = np.concatenate((points[:, :3], label[:, None]), axis=1)

    return points

# 将Carla的雷达真值转化为ndarray格式
def convert_RADAR(data, velocity):
    points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (len(data), 4))
    points = np.insert(points, 4, velocity.x ,axis = 1)
    points = np.insert(points, 5, velocity.y ,axis = 1)

    return points

# 将Carla的Bounding Box真值转化为ndarray格式
def convert_bb(bb):
    # 将列表转换为(9,)的NumPy ndarray
    bb = np.array([
        bb.location.x, bb.location.y, bb.location.z,
        bb.rotation.pitch, bb.rotation.yaw, bb.rotation.roll,
        bb.extent.x, bb.extent.y, bb.extent.z
    ])

    return bb

# 将ndarray格式的图像数据转化为jpg图片
def generate_RGB_jpg(ndarray, path):
    img = Image.fromarray(ndarray)
    img.save(path)

    return

# 将ndarray格式的图像数据转化为png图片
def generate_RGB_png(ndarray, path):
    img = Image.fromarray(ndarray)
    img.save(path)

    return

# 将ndarray格式的LIDAR数据转化为nuScenes的pcd.bin点云
def generate_LIDAR_nuScenes(ndarray, path):
    data = np.insert(ndarray, 4, np.arange(1,ndarray.shape[0]+1) ,axis = 1)
    data.tofile(path)

    return

# 将ndarray格式的LIDAR数据转化为KITTI的pcd.bin点云
def generate_LIDAR_KITTI(ndarray, path):
    data = ndarray
    data.tofile(path)

    return

# 将ndarray格式的RADAR数据转化为pcd点云
def generate_RADAR_pcd(ndarray, path):
    
    # 写文件句柄
    with open(path, 'a') as handle:

        # 获取点云点数
        point_num = ndarray.shape[0]

        # 生成pcd头部
        # nuScenes数据集RADAR数据pcd格式参考见官方文档：https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L261
        handle.write('# .PCD v0.7 - Point Cloud Data file format')
        handle.write('\nVERSION 0.7')
        handle.write('\nFIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms')
        handle.write('\nSIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1')
        handle.write('\nTYPE F F F I I F F F F F I I I I I I I I')
        handle.write('\nCOUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1')
        string = '\nWIDTH ' + str(point_num)
        handle.write(string)
        handle.write('\nHEIGHT 1')
        handle.write('\nVIEWPOINT 0 0 0 1 0 0 0')
        string = '\nPOINTS ' + str(point_num)
        handle.write(string)
        handle.write('\nDATA binary')

    # 以binary格式依次写入点
    with open(path, 'ab') as handle:

        for i in range(point_num):

            # 提取每个点的相对速度、方位角（弧度）、高度角（弧度）、距离（米）、补偿速度(x)、补偿速度(y)
            velocity = ndarray[i][0]

            # 方位角、高度角、距离
            azimuth = ndarray[i][1]
            altitude = ndarray[i][2]
            distance = ndarray[i][3]

            velocity_x_comp = ndarray[i][4]
            velocity_y_comp = ndarray[i][5]

            # 将极坐标转化为直角坐标，计算x、y、z
            x = distance * math.cos(altitude) * math.cos(azimuth)
            y = distance * math.cos(altitude) * math.sin(azimuth)
            z = distance * math.sin(altitude)

            dyn_prop = 1
            id = i
            rcs = 0.1

            vx = velocity * math.cos(altitude) * math.cos(azimuth)
            vy = velocity * math.cos(altitude) * math.sin(azimuth)

            vx_comp = velocity_x_comp
            vy_comp = velocity_y_comp

            is_quality_valid = 0
            ambig_state = 3

            x_rms = 0
            y_rms = 0

            invalid_state = 0x00
            pdh0 = 0

            vx_rms = 0
            vy_rms = 0

            bi_data = struct.pack('fffBhfffffBBBBBBBB', np.float32(x), np.float32(y), np.float32(z), np.int8(dyn_prop), np.int16(id), np.float32(rcs), 
                                  np.float32(vx), np.float32(vy), np.float32(vx_comp), np.float32(vy_comp), np.int8(is_quality_valid), 
                                np.int8(ambig_state), np.int8(x_rms), np.int8(y_rms), np.int8(invalid_state), np.int8(pdh0), np.int8(vx_rms), np.int8(vy_rms))
            
            handle.write(bi_data)
            # handle.write(b'\n')
    
    return 


def rgb_callback(image, name):
    if name == 'Front':
        queue_RGB_Front.put(image)
    elif name == 'Rear':
        queue_RGB_Rear.put(image)
    elif name == 'Left_Front':
        queue_RGB_Left_Front.put(image)
    elif name == 'Right_Front':
        queue_RGB_Right_Front.put(image)
    elif name == 'Left_Rear':
        queue_RGB_Left_Rear.put(image)
    elif name == 'Right_Rear':
        queue_RGB_Right_Rear.put(image)
    
def lidar_callback(data, sensor_name):
    queue_LIDAR_TOP.put(data)

def semantic_lidar_callback(data, sensor_name):
    queue_Semantic_LIDAR.put(data)

def radar_callback(data, sensor_name):
    queue_RADAR.put(data)

def visualize_data(lidar, text_args=(0.6)):
    
    lidar_viz = lidar_to_bev(lidar).astype(np.uint8)
    lidar_viz = cv2.cvtColor(lidar_viz,cv2.COLOR_GRAY2RGB)
 
    return lidar_viz

def lidar_to_bev(lidar, min_x=-100,max_x=100,min_y=-100,max_y=100, pixels_per_meter=4, hist_max_per_pixel=2):
    xbins = np.linspace(
        min_x, max_x+1,
        (max_x - min_x) * pixels_per_meter + 1,
    )
    ybins = np.linspace(
        min_y, max_y+1,
        (max_y - min_y) * pixels_per_meter + 1,
    )

    hist = np.histogramdd(lidar[..., :2], bins=(xbins, ybins))[0]
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel

    overhead_splat = hist / hist_max_per_pixel * 255.
    return overhead_splat[::-1,:]

if __name__ == "__main__":
    
    # 连接到客户端并检索世界对象
    client = carla.Client('localhost', 2000)
    #client.load_world(WORLD_NAME)
    world = client.get_world()

    # 获取主车
    ego_vehicle = world.get_actors().filter('vehicle.*')[0]

    # ————————————————————给主车添加RGB相机————————————————————
    camera_list = {}

    # 相机Transform和名字
    cameras_transform = [
            (carla.Transform(carla.Location(x=RGB_FRONT.get("x"), y=RGB_FRONT.get("y"), z=RGB_FRONT.get("z")),  # 前侧摄像头安装位置
                             carla.Rotation(pitch=0, yaw=RGB_FRONT.get("yaw"), roll=0)), "Front"),
            (carla.Transform(carla.Location(x=RGB_REAR.get("x"), y=RGB_REAR.get("y"), z=RGB_REAR.get("z")),  # 后侧摄像头安装位置
                             carla.Rotation(pitch=0, yaw=RGB_REAR.get("yaw"), roll=0)), "Rear"),
            (carla.Transform(carla.Location(x=RGB_SIDE_FRONT.get("x"), y=-RGB_SIDE_FRONT.get("y"), z=RGB_SIDE_FRONT.get("z")),  # 左前侧摄像头安装位置
                             carla.Rotation(pitch=0, yaw=-RGB_SIDE_FRONT.get("yaw"), roll=0)), "Left_Front"),
            (carla.Transform(carla.Location(x=RGB_SIDE_FRONT.get("x"), y=RGB_SIDE_FRONT.get("y"), z=RGB_SIDE_FRONT.get("z")),  # 右前侧的摄像头安装位置
                             carla.Rotation(pitch=0, yaw=RGB_SIDE_FRONT.get("yaw"), roll=0)), "Right_Front"),
            (carla.Transform(carla.Location(x=RGB_SIDE_REAR.get("x"), y=-RGB_SIDE_REAR.get("y"), z=RGB_SIDE_REAR.get("z")),  # 左后侧摄像头安装位置
                             carla.Rotation(pitch=0, yaw=-RGB_SIDE_REAR.get("yaw"), roll=0)), "Left_Rear"),
            (carla.Transform(carla.Location(x=RGB_SIDE_REAR.get("x"), y=RGB_SIDE_REAR.get("y"), z=RGB_SIDE_REAR.get("z")),  # 右后侧的摄像头安装位置
                             carla.Rotation(pitch=0, yaw=RGB_SIDE_REAR.get("yaw"), roll=0)), "Right_Rear")
        ]
    
    # 查找RGB相机蓝图
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

    # 设置相机参数
    camera_bp.set_attribute('fov', "90")
    camera_bp.set_attribute('image_size_x', str(IMAGE_SIZE_X))
    camera_bp.set_attribute('image_size_y', str(IMAGE_SIZE_Y))

    # 生成相机
    for index, (camera_transform, camera_position) in enumerate(cameras_transform):
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        camera_list[camera_position] = camera

    # 监听相机数据
    camera_list.get("Front").listen(lambda image: rgb_callback(image, 'Front'))
    camera_list.get("Rear").listen(lambda image: rgb_callback(image, 'Rear'))
    camera_list.get("Left_Front").listen(lambda image: rgb_callback(image, 'Left_Front'))
    camera_list.get("Right_Front").listen(lambda image: rgb_callback(image, 'Right_Front'))
    camera_list.get("Left_Rear").listen(lambda image: rgb_callback(image, 'Left_Rear'))
    camera_list.get("Right_Rear").listen(lambda image: rgb_callback(image, 'Right_Rear'))

    # ————————————————————给主车添加LIDAR————————————————————
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
 
    #设置雷达参数
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '200000')
    lidar_bp.set_attribute('range', '64')
    lidar_bp.set_attribute('rotation_frequency','20') 
    lidar_bp.set_attribute('horizontal_fov', '360') 

    # 生成雷达
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=LIDAR_HEIGHT)), attach_to=ego_vehicle)

    # 监听雷达
    lidar.listen(lambda data: lidar_callback(data, "LIDAR"))

    # ————————————————————给主车添加Semantic_LIDAR————————————————————
    semantic_lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
 
    #设置语义雷达参数
    semantic_lidar_bp.set_attribute('channels', '64')
    semantic_lidar_bp.set_attribute('points_per_second', '200000')
    semantic_lidar_bp.set_attribute('range', '64')
    semantic_lidar_bp.set_attribute('rotation_frequency','20') 
    semantic_lidar_bp.set_attribute('horizontal_fov', '360') 

    # 生成语义雷达
    semantic_lidar = world.spawn_actor(semantic_lidar_bp, carla.Transform(carla.Location(z=LIDAR_HEIGHT)), attach_to=ego_vehicle)

    # 监听语义雷达
    semantic_lidar.listen(lambda data: semantic_lidar_callback(data, "Semantic_LIDAR"))

    # ————————————————————给主车添加RADAR————————————————————
    radar_bp = world.get_blueprint_library().find('sensor.other.radar')
 
    #设置雷达参数
    radar_bp.set_attribute('horizontal_fov', '30.0')
    radar_bp.set_attribute('points_per_second', '1500')
    radar_bp.set_attribute('range', '100')
    radar_bp.set_attribute('sensor_tick','0.0') 
    radar_bp.set_attribute('vertical_fov', '30.0') 

    # 生成雷达
    radar = world.spawn_actor(radar_bp, carla.Transform(carla.Location(z=LIDAR_HEIGHT)), attach_to=ego_vehicle)

    # 监听雷达
    radar.listen(lambda data: radar_callback(data, "RADAR"))

    # ————————————————————可视化模块————————————————————

    # 为渲染实例化对象
    # init_image = np.random.randint(0, 255, (IMAGE_SIZE_Y * 2, IMAGE_SIZE_X * 3, 3), dtype='uint8')
    # surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))

    # 初始化pygame显示
    # pygame.init()
    # gameDisplay = pygame.display.set_mode((IMAGE_SIZE_X * 3, IMAGE_SIZE_Y * 2),pygame.HWSURFACE | pygame.DOUBLEBUF)

    num = 0

    crashed = False
    while not crashed:
        world.tick()

        ego_velocity = ego_vehicle.get_velocity()

        # # 获取环视相机数据
        # Front = queue_RGB_Front.get()
        # Rear = queue_RGB_Rear.get()
        # Left_Front = queue_RGB_Left_Front.get()
        # Right_Front = queue_RGB_Right_Front.get()
        # Left_Rear = queue_RGB_Left_Rear.get()
        # Right_Rear = queue_RGB_Right_Rear.get()

        # # 获取激光雷达数据
        Lidar = queue_LIDAR_TOP.get()

        # # 获取语义雷达数据
        # Semantic_Lidar = queue_Semantic_LIDAR.get()

        # 获取雷达数据
        Radar = queue_RADAR.get()

        # 获取自车边界框
        ego_transform = ego_vehicle.get_transform()
        bb = ego_vehicle.bounding_box
        bb_world = bb.get_world_vertices(ego_transform)
        bb_local = bb.get_local_vertices()

        if num == 0:
            debug = world.debug

            # ————————————————————打印传感器数据————————————————————

            # print("Front:")
            # print(type(Front))
            # print(Front)

            # print("LIDAR:")
            # print(type(Lidar))
            # print(Lidar)

            # print(Semantic_Lidar)

            # print("ego_transform:")
            # print(ego_transform)
            # print("BB:")
            # print(bb)
            # print("bb_world:")
            # for item in bb_world:
            #     print(item)
            # print("bb_local:")
            # for item in bb_local:
            #     print(item)

            # ————————————————————根据BB画线————————————————————
            # x0 = bb.location.x+ego_transform.location.x
            # y0 = bb.location.y+ego_transform.location.y
            # z0 = bb.location.z+ego_transform.location.z
            # x_len = bb.extent.x
            # y_len = bb.extent.y
            # z_len = bb.extent.z
            # thickness = 0.05
            # life_time = 10

            # p0 = carla.Location(x = x0-x_len, y = y0-y_len, z = z0-z_len)
            # p1 = carla.Location(x = x0+x_len, y = y0-y_len, z = z0-z_len)
            # p2 = carla.Location(x = x0+x_len, y = y0-y_len, z = z0+z_len)
            # p3 = carla.Location(x = x0-x_len, y = y0-y_len, z = z0+z_len)
            # p4 = carla.Location(x = x0-x_len, y = y0+y_len, z = z0-z_len)
            # p5 = carla.Location(x = x0+x_len, y = y0+y_len, z = z0-z_len)
            # p6 = carla.Location(x = x0+x_len, y = y0+y_len, z = z0+z_len)
            # p7 = carla.Location(x = x0-x_len, y = y0+y_len, z = z0+z_len)

            # debug.draw_line(p0,p1,thickness=thickness,color=carla.Color(r=255, g=0, b=0),life_time = life_time)
            # debug.draw_line(p0,p3,thickness=thickness,color=carla.Color(r=255, g=255, b=0),life_time = life_time)
            # debug.draw_line(p0,p4,thickness=thickness,color=carla.Color(r=0, g=255, b=0),life_time = life_time)
            # debug.draw_line(p1,p2,thickness=thickness,color=carla.Color(r=255, g=0, b=0),life_time = life_time)
            # debug.draw_line(p1,p5,thickness=thickness,color=carla.Color(r=0, g=255, b=0),life_time = life_time)
            # debug.draw_line(p2,p3,thickness=thickness,color=carla.Color(r=255, g=0, b=0),life_time = life_time)
            # debug.draw_line(p2,p6,thickness=thickness,color=carla.Color(r=0, g=255, b=0),life_time = life_time)
            # debug.draw_line(p3,p7,thickness=thickness,color=carla.Color(r=0, g=255, b=0),life_time = life_time)
            # debug.draw_line(p4,p5,thickness=thickness,color=carla.Color(r=0, g=0, b=255),life_time = life_time)
            # debug.draw_line(p4,p7,thickness=thickness,color=carla.Color(r=0, g=0, b=255),life_time = life_time)
            # debug.draw_line(p5,p6,thickness=thickness,color=carla.Color(r=255, g=0, b=255),life_time = life_time)
            # debug.draw_line(p6,p7,thickness=thickness,color=carla.Color(r=0, g=0, b=255),life_time = life_time)
            # debug.draw_line(carla.Location(x = x0, y = y0, z = z0),carla.Location(x = x0+50, y = y0, z = z0),thickness=thickness,color=carla.Color(r=127, g=0, b=0),life_time = life_time)
            # debug.draw_line(carla.Location(x = x0, y = y0, z = z0),carla.Location(x = x0, y = y0+50, z = z0),thickness=thickness,color=carla.Color(r=0, g=127, b=0),life_time = life_time)
            
        # Front = convert_RGB(Front)
        # Rear = convert_RGB(Rear)
        # Left_Front = convert_RGB(Left_Front)
        # Right_Front = convert_RGB(Right_Front)
        # Left_Rear = convert_RGB(Left_Rear)
        # Right_Rear = convert_RGB(Right_Rear)

        # img_combined_front = np.concatenate((Left_Front, Front, Right_Front), axis=1)
        # img_combined_rear = np.concatenate((Left_Rear, Rear, Right_Rear), axis=1)
        # img_combined = np.concatenate((img_combined_front, img_combined_rear), axis=0)

        # surface = pygame.surfarray.make_surface(img_combined.swapaxes(0, 1))

        # 按帧更新渲染的 Camera 画面
        # gameDisplay.blit(surface, (0, 0))
        # pygame.display.flip()

        Lidar = convert_LIDAR(Lidar)

        # Lidar_img = visualize_data(Lidar)

        # 可视化LIDAR
        # cv2.imshow('vizs', Lidar_img)
        # cv2.waitKey(100)

        # Semantic_Lidar = convert_Semantic_LIDAR(Semantic_Lidar)

        Radar = convert_RADAR(Radar, ego_velocity)

        # bb = convert_bb(bb)

        # ————————————————————生成并保存数据————————————————————
        # path + str(time.time()) + ".png"
        # generate_RGB_jpg(Front, "../../Carla_data/CAM_FRONT/"+ str(time.time()) + ".png")
        # generate_RGB_png(Front, "../../Carla_data/CAM_FRONT/"+ str(time.time()) + ".png")
        # generate_LIDAR_nuScenes(Lidar, "../../Carla_data/LIDAR_TOP/"+ str(time.time()) + ".pcd.bin")
        generate_LIDAR_KITTI(Lidar, "../../Carla_data/LIDAR_TOP/"+ str(time.time()) + ".pcd.bin")
        generate_RADAR_pcd(Radar, "../../Carla_data/RADAR_FRONT/"+ str(time.time()) + ".pcd")

        # if num == 0:
        #     print("Front RGB Image:")
        #     print(type(Front))
        #     print("shape:",Front.shape)
        #     print(Front)
        #     print("Lidar:")
        #     print(type(Lidar))
        #     print("shape:",Lidar.shape)
        #     print(Lidar)
        #     print("Semantic_Lidar:")
        #     print(type(Semantic_Lidar))
        #     print("shape:",Semantic_Lidar.shape)
        #     print(Semantic_Lidar)
        #     print("Radar:")
        #     print(type(Radar))
        #     print("shape:",Radar.shape)
        #     print(Radar)
        #     print("Bounding_Box:")
        #     print(type(bb))
        #     print("shape:",bb.shape)
        #     print(bb)
            
        #     num=num+1

    # 结束
    ego_vehicle.destroy()
    for camera in camera_list:
        camera.stop
    pygame.quit()



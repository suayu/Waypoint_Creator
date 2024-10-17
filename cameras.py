import random
import math
import sys
import logging
import pygame
import carla
import argparse
import numpy as np
import follow_waypoint_control
import scipy.interpolate as scipy_interpolate
from draw_line_thread import DrawInCarlaThread

WORLD_NAME = 'SUSTech_COE_ParkingLot'
WORLD_NAME = 'Town01'

PIC_SIZE = 800

PYGAME_SIZE = {
    "image_x": PIC_SIZE,
    "image_y": PIC_SIZE
}

# pygame-carla初始图像缩放参数
CAMERA_SCALING_PARAM = 25/141

CAMERA_INIT_HEIGHT = 50     # 户外场景时设置成50

# 绘图刷新间隔
REFRESH_INTERVAL =0.075

DEFAULT_POINT_HEIGHT = 0.5

# 在画线工具和Carla绘图子线程间共享数据
PARAMS = {
    'Line_points':None,
    'Camera_transform':None
}

# 视角高度变化量
VIEW_HEIGHT_VERIATION = 0.0

MIN_VIEW_HEIGHT_VERIATION = -20
MAX_VIEW_HEIGHT_VERIATION = 20

def pygame_callback(image):
    """相机传感器回调，将相机的原始数据重塑为 2D RGB，并应用于 PyGame 表面"""
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    global surface
    surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

def move_line_points(line_points, axis, value):
    """
    键盘输入控制视角移动时，对应控制绘图数据移动

    line_points --  需要移动的pygame参考点列表
    axis        --  在pygame中移动的方向
    value       --  在pygame中移动的像素数
    """
    if axis == 'x':
        line_points = [carla.Location(point.x + value/CAMERA_SCALING_PARAM, point.y, point.z) for point in line_points]
    elif axis == 'y':
        line_points = [carla.Location(point.x, point.y + value/CAMERA_SCALING_PARAM, point.z) for point in line_points]
    return line_points


def convert_to_ego_car(line_points, ego_vehicle):
    """
    将参考点列表从Carla世界坐标系转换为主车坐标系

    line_points --  需要移动的pygame参考点列表
    ego_vehicle --  自车
    points_of_ego_vehicle_coordinate_system --  基于自车坐标系的参考点位置列表(x,y)
    """
    ego_car_matrix = ego_vehicle.get_transform().get_matrix()
    ego_car_matrix = np.mat(ego_car_matrix)
    car_matrix = ego_car_matrix.I
    points_of_ego_vehicle_coordinate_system = []

    for point in line_points:
        car_P = np.dot(car_matrix, point)
        car_P = car_P[:2]
        points_of_ego_vehicle_coordinate_system.append((car_P[0,0],car_P[0,1]))

    return points_of_ego_vehicle_coordinate_system

def save_points(points, filename, axis_num):
    with open(filename, 'w') as file:
        for point in points:
            if axis_num == 3:
                file.write(f"{point[0]}, {point[1]}, {point[2]}\n")
            if axis_num == 2:
                file.write(f"{point[0]}, {point[1]}\n")

def save_yaws(yaws, points, filename):
    with open(filename, 'w') as file:
        for i in range(len(yaws)): 
            file.write(f"{points[i].x}, {points[i].y}, {points[i].z}, {yaws[i]}\n")
        length = len(points) - 1
        file.write(f"{points[length].x}, {points[length].y}, {points[length].z}\n")

def calculate_yaws(points):
    yaws = []
    for i in range(len(points) - 1): 
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
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

def interpolate_b_spline_path(x, y, n_path_points, degree=3):
    """
    调用B样条函数实现曲线平滑

    x   --  曲线x坐标列表
    y   --  曲线y坐标列表
    n_path_points   --  插值曲线采样点数量
    degree          --  B样条曲线阶数
    """
    ipl_t = np.linspace(0.0, len(x) - 1, len(x))
    l,r=[(2,0.0)],[(2,0.0)]
    spl_i_x = scipy_interpolate.make_interp_spline(ipl_t, x, k=degree,bc_type=(l,r))
    spl_i_y = scipy_interpolate.make_interp_spline(ipl_t, y, k=degree,bc_type=(l,r))
    travel = np.linspace(0.0, len(x) - 1, n_path_points)
    return spl_i_x(travel), spl_i_y(travel)

def interpolate_path(path, sample_rate, times):
    if len(path) == 0:
        return path
    choices = np.arange(0,len(path),sample_rate)
    if len(path)-1 not in choices:
        choices =  np.append(choices , len(path)-1)

    way_point_x = [path[index].x for index in choices]
    way_point_y = [path[index].y for index in choices]

    for i in range(times):
        way_point_x, way_point_y = interpolate_b_spline_path(way_point_x, way_point_y, len(path))
    new_path = np.vstack([way_point_x,way_point_y]).T
    return new_path

def altitude_check(points, threshold_dz = 1):
    if len(points) < 3:
        return points

    # 推导式写法
    # smoothed_points = [
    #     carla.Location(x = points[i].x, y = points[i].y, z = (points[i - 1].z + points[i + 1].z) / 2)
    #     if i > 0 and i < len(points) - 1 and (
    #         abs(points[i].z - points[i - 1].z) > threshold_dz and
    #         abs(points[i].z - points[i + 1].z) > threshold_dz
    #     )
    #     else points[i]
    #     for i in range(len(points))
    # ]
    # return smoothed_points

    for i in range(1, len(points) - 1):

        interpolation_with_prev = abs(points[i].z - points[i-1].z)
        interpolation_with_next = abs(points[i].z - points[i+1].z)
        # print("interpolation_with_prev:",interpolation_with_prev," interpolation_with_next:",interpolation_with_next)
        if interpolation_with_prev > threshold_dz and interpolation_with_next > threshold_dz:
            points[i].z = (points[i-1].z + points[i+1].z) / 2

    return points


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)

    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_id', type=int, default=-1, help='ID of actor')
    parser.add_argument('--use_z', type=bool, default=False, help='Whether to enable Z-coordinate')
    args = parser.parse_args()
    actor_id = args.actor_id
    use_z = args.use_z

    # 连接到客户端并检索世界对象
    client = carla.Client('localhost', 2000)
    client.load_world(WORLD_NAME)
    world = client.get_world()

    # 地下车库高度：-2.2～2.6

    # 获取车辆并设置自动驾驶
    if actor_id == -1:

        # 获取地图的刷出点
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle_bp = world.get_blueprint_library().filter('*vehicle*').filter('vehicle.tesla.*')[0]
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        # ego_vehicle.set_autopilot(True)
    else:
        actor_list = world.get_actors()
        actor = actor_list.find(actor_id)
        if actor == None:
            logging.warning("ERROR ACTOR ID!")
            sys.exit()
        else:
            ego_vehicle = actor_list.find(actor_id)

    world.get_spectator().set_transform(carla.Transform(ego_vehicle.get_transform().location+carla.Location(z=CAMERA_INIT_HEIGHT),carla.Rotation(pitch=-90)))

    # 生成摄像头
    image_size_x = int(PYGAME_SIZE.get("image_x"))
    image_size_y = int(PYGAME_SIZE.get("image_y"))
    camera_transform = carla.Transform(carla.Location(x=ego_vehicle.get_transform().location.x,y=ego_vehicle.get_transform().location.y,z=CAMERA_INIT_HEIGHT), 
                                       carla.Rotation(pitch=-90.0, yaw=0, roll=0))
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('fov', "110")
    camera_bp.set_attribute('image_size_x', str(image_size_x))
    camera_bp.set_attribute('image_size_y', str(image_size_y))
    camera = world.spawn_actor(camera_bp, camera_transform)

    # 采集carla世界中camera的图像
    camera.listen(lambda image: pygame_callback(image))
    camera_transform = camera.get_transform()

    # 将相机图像加载到pygame表面
    init_image = np.random.randint(0, 255, (PYGAME_SIZE.get("image_y"), PYGAME_SIZE.get("image_x"), 3), dtype='uint8')
    surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))

    # 初始化pygame显示
    pygame.init()
    gameDisplay = pygame.display.set_mode((PYGAME_SIZE.get("image_x"), PYGAME_SIZE.get("image_y")),pygame.HWSURFACE | pygame.DOUBLEBUF)

    line_points = []
    drawing = False

    draw_thread = DrawInCarlaThread(world, REFRESH_INTERVAL, PIC_SIZE, CAMERA_SCALING_PARAM, CAMERA_INIT_HEIGHT, DEFAULT_POINT_HEIGHT, use_z)
    draw_thread.start()

    crashed = False

    while not crashed:

        # 等待同步
        world.tick()

        # 按帧更新渲染的 Camera 画面
        gameDisplay.blit(surface, (0, 0))

        # 获取 pygame 事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

            key_list = pygame.key.get_pressed()
            if key_list[pygame.K_w]:
                camera_transform.location.x += 1
                line_points = move_line_points(line_points,'y',1)
            if key_list[pygame.K_s]:
                camera_transform.location.x -= 1
                line_points = move_line_points(line_points,'y',-1)
            if key_list[pygame.K_a]:
                camera_transform.location.y -= 1
                line_points = move_line_points(line_points,'x',1)
            if key_list[pygame.K_d]:
                camera_transform.location.y += 1
                line_points = move_line_points(line_points,'x',-1)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    carla_points = draw_thread.convert_PIL_points_to_carla(line_points, camera_transform)
                    save_points(carla_points, "point_in_carla.txt", 3)
                    ego_car_points = convert_to_ego_car(carla_points, ego_vehicle)
                    save_points(ego_car_points, "point_in_ego_car_system.txt", 2)
                    logging.info("data saved")
                elif event.key == pygame.K_h:
                    VIEW_HEIGHT_VERIATION = 0
                    camera_transform.location.z = CAMERA_INIT_HEIGHT
                elif event.key == pygame.K_p:
                    line_points = []
                elif event.key == pygame.K_y:
                    carla_points = draw_thread.convert_PIL_points_to_carla(line_points, camera_transform)
                    yaws = calculate_yaws(carla_points)
                    save_yaws(yaws,line_points,"yaws.txt")
                    logging.info("yaws saved")
                elif event.key == pygame.K_b:

                    interpolated_path = interpolate_path(line_points, sample_rate = 5, times = 3)

                    final_interpolated_path = []
                    for point in interpolated_path:
                        z = draw_thread.get_z_coordinate(point, camera_transform)
                        new_point = carla.Location(point[0], point[1], z)
                        final_interpolated_path.append(new_point)

                    line_points = final_interpolated_path
                    line_points = altitude_check(line_points)
                elif event.key == pygame.K_e:

                    # 前进启动追踪
                    reference_points = draw_thread.get_reference_points()
                    follow_waypoint_control.main(forward = True, create_ego_car = False, reference_points = reference_points)

                elif event.key == pygame.K_q:

                    # 倒车启动追踪
                    reference_points = draw_thread.get_reference_points()
                    follow_waypoint_control.main(forward = False, create_ego_car = False, reference_points = reference_points)

            # 绘制线条
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                line_points = altitude_check(line_points)
            elif event.type == pygame.MOUSEMOTION and drawing:
                new_pos = ((event.pos[0] - PIC_SIZE/2)*((CAMERA_INIT_HEIGHT + VIEW_HEIGHT_VERIATION)/CAMERA_INIT_HEIGHT) + PIC_SIZE/2 , 
                           (event.pos[1] - PIC_SIZE/2)*((CAMERA_INIT_HEIGHT + VIEW_HEIGHT_VERIATION)/CAMERA_INIT_HEIGHT) + PIC_SIZE/2)
                z = draw_thread.get_z_coordinate(new_pos, camera_transform)
                point_xyz = carla.Location(new_pos[0], new_pos[1], z)
                line_points.append(point_xyz)
            elif event.type == pygame.MOUSEWHEEL:

                # 根据鼠标滚轮的方向调整缩放因子
                if event.y == -1:  
                    VIEW_HEIGHT_VERIATION += 0.5
                else:  
                    VIEW_HEIGHT_VERIATION -= 0.5
                VIEW_HEIGHT_VERIATION = max(MIN_VIEW_HEIGHT_VERIATION, min(VIEW_HEIGHT_VERIATION, MAX_VIEW_HEIGHT_VERIATION))
                camera_transform.location.z = CAMERA_INIT_HEIGHT + VIEW_HEIGHT_VERIATION

            draw_thread.update_line_points(line_points)
            draw_thread.update_camera_transform(camera_transform)
            camera.set_transform(camera_transform)

        pygame.display.flip()

    ego_vehicle.destroy()
    camera.stop()
    draw_thread.stop()

    pygame.quit()
import threading
import time
import carla
import numpy as np

class DrawInCarlaThread:
    def __init__(self, world: carla.World, interval, pic_size, camera_scaling_param, camera_init_height, default_point_height, use_z):
        self.world = world
        self.lock = threading.Lock()
        self.thread_hold = True
        self.line_points = None
        self.camera_transform = None
        self.interval = interval
        self.pic_size = pic_size
        self.camera_scaling_param = camera_scaling_param
        self.camera_init_height = camera_init_height
        self.default_point_height = default_point_height
        self.use_z = use_z

    def draw_in_carla_thread(self):
        """实现绘图进程"""
        while self.thread_hold:
            with self.lock:
                camera_transform = self.camera_transform
                line_points = self.line_points
            if not (line_points is None or len(line_points) == 0 or camera_transform is None):
                carla_points = self.convert_PIL_points_to_carla(line_points, camera_transform)
                self.draw_in_carla(carla_points)
            time.sleep(self.interval)

    def get_reference_points(self):
        return self.convert_PIL_points_to_carla(self.line_points, self.camera_transform)

    def start(self):
        self.thread = threading.Thread(target=self.draw_in_carla_thread)
        self.thread_hold = True
        self.thread.start()

    def stop(self):
        self.thread_hold = False
        self.thread.join()

    def convert_PIL_points_to_carla(self, line_points, camera_transform):

        # 将pygame坐标系转化为Carla坐标系
        result_points = []
        for point in line_points:
            x_0 = point.x
            y_0 = point.y
            x_1 = self.pic_size - y_0 - self.pic_size/2
            y_1 = x_0 - self.pic_size/2
            A = np.array([x_1,y_1])

            # 计算缩放矩阵
            s_x = self.camera_scaling_param
            s_y = self.camera_scaling_param

            scale_matrix = np.array([[s_x, 0], 
                                    [0, s_y]])

            # 计算平移矩阵
            Location = camera_transform.location
            t_x = Location.x
            t_y = Location.y
            translation_matrix = np.array([[1, 0, t_x], 
                                            [0, 1, t_y], 
                                            [0, 0, 1]])

            # 计算旋转矩阵
            angle = np.radians(camera_transform.rotation.yaw)
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                        [np.sin(angle), np.cos(angle)]])

            # 缩放变换
            scaled_A = np.dot(scale_matrix, A)

            # 平移变换
            translated_A = np.dot(translation_matrix, np.append(scaled_A, 1))
            translated_A = translated_A[:2] 

            # 旋转变换
            rotated_A = np.dot(rotation_matrix, translated_A)

            rotated_A = np.append(rotated_A, point.z)
            rotated_A = np.append(rotated_A, 1)
            
            result_points.append(rotated_A)

        return result_points

    def draw_in_carla(self, reference_points):
        debug = self.world.debug

        # 绘制参考点
        for point in reference_points:
            if self.use_z:
                debug.draw_point(carla.Location(x=point[0], y=point[1], z=point[2]), size=0.08, color=carla.Color(r=255, g=0, b=0), life_time=self.interval*2)
            else:
                debug.draw_point(carla.Location(x=point[0], y=point[1], z=0.5), size=0.08, color=carla.Color(r=255, g=0, b=0), life_time=self.interval*2)
        
        # 绘制连线
        for i in range(len(reference_points) - 1):
            if self.use_z:
                debug.draw_line(
                    carla.Location(x=reference_points[i][0], y=reference_points[i][1], z=reference_points[i][2]),
                    carla.Location(x=reference_points[i + 1][0], y=reference_points[i + 1][1], z=reference_points[i + 1][2]),
                    thickness=0.1,
                    color=carla.Color(r=255, g=0, b=0),
                    life_time = self.interval*2
                )
            else:
                debug.draw_line(
                    carla.Location(x=reference_points[i][0], y=reference_points[i][1], z=0.5),
                    carla.Location(x=reference_points[i + 1][0], y=reference_points[i + 1][1], z=0.5),
                    thickness=0.1,
                    color=carla.Color(r=255, g=0, b=0),
                    life_time = self.interval*2
                )

    def update_line_points(self, line_points):
        with self.lock:
            self.line_points = line_points

    def update_camera_transform(self, camera_transform):
        with self.lock:
            self.camera_transform = camera_transform

    def is_label_valid(self, label):
        """
        判断语义标签是否合法

        label   --  Carla语义标签
        invalid_labels  --  非法语义列表

        完整语义标签对照表可参考网址 : https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
        其中，
        6 : Poles
        7 : TrafficLight
        8 : TrafficSign
        如果label为invalid_labels中的任何一种,则说明标签不合法
        """
        
        invalid_labels = {
            6,
            7,
            8,
        }
        return label not in invalid_labels

    def get_z_coordinate(self, point, camera_transform):
        """
        计算pygame坐标系下的点在Carla中的对应点高度

        point               --  pygame坐标系下的点(x,y)
        camera_transform    --  相机transform矩阵
        z                   --  Carla中对应点高度
        """

        points = []
        points.append(carla.Location(x=point[0], y=point[1], z=self.default_point_height))
        result = self.convert_PIL_points_to_carla(points, camera_transform)
        result_point = result[0]

        # Compute the z coordinate
        point_lists = self.world.cast_ray(carla.Location(result_point[0], result_point[1], self.camera_init_height),carla.Location(result_point[0], result_point[1], -5))
        z = self.default_point_height
        if len(point_lists) != 0:
            for point in point_lists:
                if self.is_label_valid(point.label):
                    z = point.location.z + self.default_point_height
                    break
        return z
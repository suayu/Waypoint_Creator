import argparse
import numpy as np

def read_data(read_path):
    data = np.load(read_path, allow_pickle=True)
    data = data.reshape(-1, 10)

    transform = data[:,:6]
    time_stamp = data[:,-4:]

    return transform, time_stamp

def interpolate(transform, time_stamp, dT):

    time_stamp = time_stamp[:,1]
    time_stamp[:] -= time_stamp[0]

    len = transform.shape[0]
    time_end = time_stamp[len-1]

    new_len = int(time_end / dT)
    new_time_stamp = np.arange(0, new_len * dT, dT)

    x = transform[:,0]
    y = transform[:,1]
    z = transform[:,2]
    pitch = transform[:,3]
    yaw = transform[:,4]
    roll = transform[:,5]

    x_interpolated = np.interp(new_time_stamp, time_stamp, x)
    y_interpolated = np.interp(new_time_stamp, time_stamp, y)
    z_interpolated = np.interp(new_time_stamp, time_stamp, z)
    pitch_interpolated = np.interp(new_time_stamp, time_stamp, pitch)
    yaw_interpolated = np.interp(new_time_stamp, time_stamp, yaw)
    roll_interpolated = np.interp(new_time_stamp, time_stamp, roll)

    # for i in range(0,new_len):
    #     print("timestamp:",format(time_stamp[i], '.4f'),"  new timestamp:",format(new_time_stamp[i], '.2f'),"  x,y,z:",format(x[i], '.4f'),format(y[i], '.4f'),
    #           format(z[i], '.4f'),"  pitch,yaw,roll:",format(pitch[i], '.4f'),format(yaw[i], '.4f'),format(roll[i], '.4f'),"  new x,y,z:",
    #           format(x_interpolated[i], '.4f'),format(y_interpolated[i], '.4f'),format(z_interpolated[i], '.4f'),"  new pitch,yaw,roll:",
    #           format(pitch_interpolated[i], '.4f'),format(yaw_interpolated[i], '.4f'),format(roll_interpolated[i], '.4f'))

    # for i in range(new_len,len):
    #     print("timestamp:",format(time_stamp[i], '.4f'),"  x,y,z:",format(x[i], '.4f'),format(y[i], '.4f'),
    #           format(z[i], '.4f'),"  pitch,yaw,roll:",format(pitch[i], '.4f'),format(yaw[i], '.4f'),format(roll[i], '.4f'))

    return x_interpolated, y_interpolated, z_interpolated, pitch_interpolated, yaw_interpolated, roll_interpolated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Carla Ego Car Transform Reader')
    parser.add_argument('-pth', '--read_path', default = '')
    parser.add_argument('-dt', '--dt', type = float, default = 0.1)
    args = parser.parse_args()

    transform, time_stamp = read_data(args.read_path)
    x, y, z, pitch, yaw, roll = interpolate(transform, time_stamp, args.dt)



    
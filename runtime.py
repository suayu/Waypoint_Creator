import numpy as np

lam = 0.2
L = 4.524
H = 2.07642
c = 2
dt = 0.1

x_lim = 500
y_lim = 500

save_fig_dir = './pic'
save_fig_name = 'res.png'

# BB:
# BoundingBox(Location(x=0.137793, y=0.000000, z=1.038210), Extent(x=2.399997, y=1.082433, z=1.038210), Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
# bb_world:
# Location(x=211.618240, y=58.820248, z=0.040095)
# Location(x=211.618240, y=58.820248, z=2.116515)
# Location(x=211.618240, y=60.985111, z=0.040095)
# Location(x=211.618240, y=60.985111, z=2.116515)
# Location(x=216.418243, y=58.820248, z=0.040095)
# Location(x=216.418243, y=58.820248, z=2.116515)
# Location(x=216.418243, y=60.985111, z=0.040095)
# Location(x=216.418243, y=60.985111, z=2.116515)
# bb_local:
# Location(x=-2.262204, y=-1.082433, z=0.000000)
# Location(x=-2.262204, y=-1.082433, z=2.076420)
# Location(x=-2.262204, y=1.082433, z=0.000000)
# Location(x=-2.262204, y=1.082433, z=2.076420)
# Location(x=2.537790, y=-1.082433, z=0.000000)
# Location(x=2.537790, y=-1.082433, z=2.076420)
# Location(x=2.537790, y=1.082433, z=0.000000)
# Location(x=2.537790, y=1.082433, z=2.076420)

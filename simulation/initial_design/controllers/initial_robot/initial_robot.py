"""initial_robot controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Motor, Camera, DistanceSensor, LED, LightSensor

# create the Robot instance.
TIME_STEP = 64
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

ds = []
dsNames = ['ds_right', 'ds_left']

for i in range(2):
    ds.append(robot.getDevice(dsNames[i]))
    ds[i].enable(TIME_STEP)
wheels = []
wheelsNames = ['wheel1', 'wheel2', 'wheel3', 'wheel4']
for i in range(4):
    wheels.append(robot.getDevice(wheelsNames[i]))
    wheels[i].setPosition(float('inf'))
    wheels[i].setVelocity(0.0)
avoidObstacleCounter = 0

lr = robot.getDevice("linear")
lr.setVelocity(0.1)  # optional, ensures smooth movement
lr.setPosition(0.0)  # initial reset

rm = robot.getDevice("camera_rm")

cam = robot.getDevice("camera")
cam.enable(timestep)

sensor_disc = robot.getDevice("linear_sensor")

light_sensor = robot.getDevice("light_sensor")
light_sensor.enable(timestep)

led = robot.getDevice("led")

i = 0

linear = 0
rotate = 0
cameraStart = False
leftDone = False
rightDone = False
cameraFinish = False

moistureStart = False
moistureEnd = False
linear_s = 0

WALK_STEP = 400
# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    if i%WALK_STEP == 0:
        wheels[0].setVelocity(0)
        wheels[1].setVelocity(0)
        wheels[2].setVelocity(0)
        wheels[3].setVelocity(0)
        cameraStart = True
        moistureStart = True
    
    if moistureStart:
        led.set(1)
        linear_s -= 0.002
        
        if linear_s <= -0.09:
            linear_s = -0.09
            print("Got moisture and PNK data")
            moistureStart = False
            moistureEnd = True
        
        sensor_disc.setPosition(linear_s)
        i += 1
        
    elif moistureEnd:
        linear_s += 0.002
        
        if linear_s > 0:
            linear_s = 0
            led.set(0)
            moistureEnd = False
            moistureStart = False
        
        sensor_disc.setPosition(linear_s)
            
    
    if cameraStart:
        if linear < 0.29:
            linear += 0.005
        elif linear > 0.29:
            if leftDone and rotate < 1.55 and not rightDone:
                rotate += 0.05
            elif rotate < 0.08 and rotate > -0.08 and leftDone and rightDone:
                leftDone = False
                rightDone = False
                cameraStart = False
                cameraFinish = True
                i += 1
            elif leftDone and rightDone:
                rotate -= 0.05
            elif rotate < 0.08 and rotate > -0.08 and not leftDone and not rightDone:
                print("Saving front view")
                cam.saveImage("img_"+str(i//WALK_STEP)+"_front.png", 100)
                rotate -= 0.05
            elif rotate > 1.55:
                rightDone = True
                print("Saving right view")
                cam.saveImage("img_"+str(i//WALK_STEP)+"_right.png", 100)
                rotate -= 0.05
            elif rotate > -1.55:
                rotate -= 0.05
            elif rotate < -1.55:
                leftDone = True
                print("Saving left view")
                cam.saveImage("img_"+str(i//WALK_STEP)+"_left.png", 100)
                rotate += 0.05
            
        lr.setPosition(linear)
        rm.setPosition(rotate)
    
    elif cameraFinish:
        linear -= 0.005
        if linear < 0:
            linear = 0
            cameraFinish = False
            i += 1
        
        lr.setPosition(linear)
    else:
        leftSpeed = 1.0
        rightSpeed = 1.0
        if avoidObstacleCounter > 0:
            avoidObstacleCounter -= 1
            leftSpeed = 1.0
            rightSpeed = -1.0
        else:  # read sensors
            for j in range(2):
                if ds[j].getValue() < 950.0:
                    avoidObstacleCounter = 80
        wheels[0].setVelocity(leftSpeed)
        wheels[1].setVelocity(rightSpeed)
        wheels[2].setVelocity(leftSpeed)
        wheels[3].setVelocity(rightSpeed)
        i = i + 1
    
    if i == 2969:
        wheels[0].setVelocity(0)
        wheels[1].setVelocity(0)
        wheels[2].setVelocity(0)
        wheels[3].setVelocity(0)
        break
    # print("Camera Position")
    # print("Linear Position:", linear)
    # print("Rotation Position:", rotate)
    # print("Sensor Position:", linear_s)
    # print("LED:", led.get())
    # print("Light Sensor Value:", light_sensor.getValue())
    
cam.disable()
light_sensor.disable()
# Enter here exit cleanup code.

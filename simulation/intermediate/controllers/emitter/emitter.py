from controller import Robot

# Initialize the Robot instance
robot = Robot()

# Get the simulation time step
timestep = int(robot.getBasicTimeStep())

# Get the Emitter device
emitter = robot.getDevice('emitter')

# Set the channel to 100 as per your requirement
emitter.setChannel(100)

# We use the node's name (A1, A2, or A3) as the message
# This tells the receiver which anchor is sending the signal
anchor_id = robot.getName()
message = anchor_id.encode('utf-8')

print(f"Anchor {anchor_id} started broadcasting on channel 100...")

while robot.step(timestep) != -1:
    # Continuously broadcast the ID
    # The Receiver will use the strength of this signal to calculate distance
    emitter.send(message)
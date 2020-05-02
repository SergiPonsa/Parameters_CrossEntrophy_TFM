import sys
import pybullet as p
import numpy as np
import time
sys.path.insert(1,"../Simulation_Pybullet/")
from KinovaGen3Class import KinovaGen3

class env_pybullet_kin_gen3:
    "Enviorment of pybullet for the Kinova Gen3 "


class env_pybullet_kin_gen3() :

    def __init__(self,visual=True):
        self.visual = visual
        if (self.visual == True):
            p.connect(p.GUI)
            print("hola")
        else:
            p.connect()
        p.setGravity(0.0, 0.0, -9.81)
        self.robot = KinovaGen3(robot_urdf="../Simulation_Pybullet/models/urdf/JACO3_URDF_V11.urdf")
        print("hola")
        #Get dynamics info
        Data = self.robot.get_robot_pybullet_param_dynamics(self.robot.robot_control_joints)
        print(Data.shape)
        #Get dynamics parameters by variable
        [self.mass,self.lateral_friction,self.inertia,self.inertial_pos,self.inertial_orn,\
                                self.restitution,self.rolling_friction,self.spinning_friction,\
                                self.contact_damping,self.contact_stiffness,\
                                self.body_type,self.collision_margin] = np.hsplit(Data,Data.shape[1])
        #pass to only 1 value for varialble
        #print(self.inertia)
        self.inertia =self.inertia[:,0]
        #print(self.inertia)
        #print(self.inertia.shape)
        inertia = []
        for element in self.inertia:
            inertia.append(list(element))
        self.inertia =np.asarray(inertia)
        #print(self.inertia)
        #print(self.inertia.shape)

        self.Ixx,self.Iyy,self.Izz = np.hsplit(self.inertia,self.inertia.shape[1])
        #print (self.Ixx)
        #print (self.Iyy)
        #print (self.Izz)

        #Control parameters
        self.kp,self.ki,self.kd=[0.1,0,0]
        self.max_vel,self.force_x_one = [30,0.5]


        #Get the robot joints jointInfo

        Data = self.robot.get_robot_pybullet_param_joints(self.robot.robot_control_joints)
        print(Data.shape)
        #Get dynamics parameters by variable
        [self.joint_id,self.joint_name,self.joint_type,self.q_pos,self.q_vel,\
                                self.flags,self.damping,self.friction,\
                                self.lower,self.upper,self.effort,\
                                self.velocity,self.link_name,self.joint_axis,\
                                self.parent_frame_pos,self.parent_frame_orn,self.parent_link_id] = np.hsplit(Data,Data.shape[1])

        self.original_parameters = [self.mass,self.Ixx,self.Iyy,self.Izz]
        self.observation_space = 1
        self.action_space = len(list(self.original_parameters))

        print(self.link_name)
        print(self.joint_id)
        print(self.parent_link_id)

    def reset(self):
        p.disconnect() #disconnect pybullet
        if (self.visual == True):
            p.connect(p.GUI)
        else:
            p.connect(p.GUI)
        self.robot = KinovaGen3()
        state = 1
        return state

    def step(self,action,perone=True):
        if (perone == True):
            self.action = self.original_parameters *self.action

        #I abstract the data again to individual variables
        mass,Ixx,Iyy,Izz,force = self.action
        inertia = [Ixx,Iyy,Izz]
        self.robot.changeparameters(["mass","inertia"],[mass,inertia])
        reward = Do_experiment (kp,ki,kd,max_vel,self_force_x_one)
        state,reward,done,_ = [1,reward,True,None]
        return  [state,reward,done,_]

if (__name__=="__main__"):
    env = env_pybullet_kin_gen3()
    while(True):
        time.sleep(0.01)

import sys
import pybullet as p
import numpy as np
import pandas as pd
import time
sys.path.insert(1,"../Simulation_Pybullet/")
from KinovaGen3Class import KinovaGen3
from Rotations import *

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
        self.kp = np.array([0.1]*self.robot.number_robot_control_joints)
        self.ki = np.array([0.0]*self.robot.number_robot_control_joints)
        self.kd = np.array([0.0]*self.robot.number_robot_control_joints)
        self.max_vel = np.array([30]*self.robot.number_robot_control_joints)
        self.force_x_one = np.array([1]*self.robot.number_robot_control_joints)


        #Get the robot joints jointInfo

        Data = self.robot.get_robot_pybullet_param_joints(self.robot.robot_control_joints)
        print(Data.shape)
        #Get dynamics parameters by variable
        [self.joint_id,self.joint_name,self.joint_type,self.q_pos,self.q_vel,\
                                self.flags,self.damping,self.friction,\
                                self.lower,self.upper,self.effort,\
                                self.velocity,self.link_name,self.joint_axis,\
                                self.parent_frame_pos,self.parent_frame_orn,self.parent_link_id] = np.hsplit(Data,Data.shape[1])


        #Create a data frame with all the data original, to study

        self.original_parameters_df = pd.DataFrame({})
        self.original_parameters_df ["mass"] = self.mass
        self.original_parameters_df ["damping"] = self.damping
        self.original_parameters_df ["Ixx"] = self.Ixx
        self.original_parameters_df ["Iyy"] = self.Iyy
        self.original_parameters_df ["Izz"] = self.Izz
        self.original_parameters_df ["Izz"] = self.Izz
        self.original_parameters_df ["kp"] = self.kp
        self.original_parameters_df ["ki"] = self.ki
        self.original_parameters_df ["kd"] = self.kd
        self.original_parameters_df ["max_vel"] = self.max_vel
        self.original_parameters_df ["force_x_one"] = self.force_x_one

        self.modified_parameters_df = self.original_parameters_df.copy(deep=True)

        #Has to exist in the data frame , and be equal to the column name
        #The action must be introduced in the same order, and as many values as joints are expected
        self.parameters_to_modify = ["mass"]

        self.observation_space = 1
        self.action_space = len(self.parameters_to_modify)*self.robot.number_robot_control_joints

        #print(self.link_name)
        #print(self.joint_id)
        #print(self.parent_link_id)
        #print(self.damping)

    def reset(self):
        p.disconnect() #disconnect pybullet
        if (self.visual == True):
            p.connect(p.GUI)
        else:
            p.connect(p.GUI)
        self.robot = KinovaGen3()
        state = 1
        return state

    def step(self,action):
        #Check the length it's right
        if(len(list(action)!=self.parameters_to_modify*self.robot.number_robot_control_joints):
        #I abstract the data again to individual variables
        parameters_value = np.split(action,len(self.parameters_to_modify))

        #I modify the corresponding parameters in the dataframe
        for i in range( len(self.parameters_to_modify) ):
            self.modified_parameters_df[self.parameters_to_modify[i]] = \
                parameters_value[i]

        #Modify robot

        aux = np.vstack(( list(self.modified_parameters_df["Ixx"]),\
                        list(self.modified_parameters_df["Iyy"]),
                        list(self.modified_parameters_df["Izz"]) ))
        aux = aux.T

        aux = aux.reshape(1,aux.size)

        inertia = list(list(aux).pop())

        self.robot.modify_robot_pybullet(self.robot.robot_control_joints,\
                                        ["mass","damping","inertia"],\
                                        [list(self.modified_parameters_df["mass"]),\
                                        list(self.modified_parameters_df["damping"]),\
                                        inertia]\
                                         )
        self.robot = self.Do_Experiment (repeats,experiment,kp,ki,kd,max_vel,self_force_x_one)
        df_test = self.Do_Average_experiments()
        reward = self.Compute_Reward(df_test,df_Okay)
        state,reward,done,_ = [1,reward,True,None]
        return  [state,reward,done,_]

    def Do_Experiment(self,repeats,experiment,robot=None,max_vel_list=[30],force_per_one_list=[1],joint = 1,kp_list=[0.1],ki_list=[0.0],kd_list=[0.0]):
        if(robot == None):
            robot = self.robot

        if (len(max_vel_list) == 1):
            max_vel_list = max_vel_list * robot.number_robot_control_joints
        if (len(force_per_one_list) == 1):
            force_per_one_list = force_per_one_list * robot.number_robot_control_joints
        if (len(kp_list) == 1):
            kp_list = kp_list * robot.number_robot_contrkd_listol_joints
        if (len(ki_list) == 1):
            ki_list = ki_list * robot.number_robot_control_joints
        if (len(kd_list) == 1):
            kd_list = kd_list * robot.number_robot_control_joints

        for iteration in range(repeats):
            # Initialization
            counter = simSteps(experiment,timestep) # detemine time

            #create PIDs
            PID_List = []
            for i in range(robot.number_robot_control_joints ):
                PID_List.append( PID(max_velocity=max_vel_list[i],kp=kp_list[i],ki=ki_list[i],kd=kd_list[i]) )
            robot.database_name = "Data_"+str(iteration)

            #Move to joint 0
            angles_zero = [0.0]*robot.number_robot_control_joints
            print(angles_zero)
            robot.save_database = False
            robot.move_joints(joint_param_value = angles_zero, wait=True,desired_force_per_one=force_per_one_list)

            #Start saving data every time step
            robot.save_database = True
            #time.sleep(10)
            #Execute experiment during the especified time
            for simStep in range(counter):

                #Every step compute the pid
                PID_List = set_target_thetas(counter, PID_List,experiment,"PyBullet",simStep,joint)
                print
                #Every 12 steeps apply the control
                if simStep % 12 == 0:
                    current_angles=robot.get_actual_control_joints_angle()
                    velocity_angles= []

                    #Compute the velocity
                    for i in range( len(self.robot.robot_control_joints ) ):
                        print(i)
                        print(PID_List[i].get_target_theta())
                        print(current_angles)
                        velocity_angles.append( PID_List[i].get_velocity(math.degrees(current_angles[i])) /57.32484076 )

                    #Apply the controls
                    robot.move_joints_control_vel( joint_param_value = velocity_angles ,wait = False , desired_force_per_one_list=force_per_one_list)

                robot.step_simulation()

        #Change one lasttime the name and simulate to append it to the database list
        robot.database_name = "dummy"
        robot.step_simulation()

        return robot

    def Do_Average_experiments(self,robot = None):
        if(robot == None):
            robot = self.robot

        dataframe_list = []
        angles_list = []

        #Go through all the database classes created during the experiment
        for data in robot.database_list:

            #pass the data to a list to do the average
            angles_list.append(data.joints_angles_rad)

            print(data.name)
            print("\n")

        #compute the average, convert all the data to numpy to make it easier
        joint_angles_array = np.array(angles_list)
        print(joint_angles_array.shape)

        #dimensions of the numpy
        [experiments,steps,joints] = joint_angles_array.shape

        average_steps = []
        for stp in range(steps):
            average_joints = []
            for j in range(joints):
                average_value = np.average(joint_angles_array[:,stp,j])
                average_joints.append(average_value)
            average_steps.append(average_joints)

        #create the average data frame
        avg_df = pd.DataFrame({})

        joint_angles_average = np.array(average_steps)
        for i in range(len(robot.robot_control_joints)):
            column_name = "joint"+str(i)

            avg_df[column_name] = joint_angles_average [:,i]
        avg_df.index = data.time

        return avg_df

    def Compute_Reward(df_test,df_Okay=None):
        if(df_Okay == None):
            df_Okay = self.df_Okay

        np_Substract = df_test.to_numpy()
        np_ToSubstract = df_Okay.to_numpy()

        if (np_Substract.shape != np_ToSubstract.shape):

            print("Data samples have diferent shapes correct it")

        np_result = np_ToSubstract[:,1:] - np_Substract[:,1:]
        np_result = np.absolute(np_result) *-1
        reward = np_result.sum()

        return reward
if (__name__=="__main__"):
    env = env_pybullet_kin_gen3()
    while(True):
        time.sleep(0.01)

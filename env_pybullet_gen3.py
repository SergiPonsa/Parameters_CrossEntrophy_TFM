import sys
import pybullet as p
import numpy as np
import pandas as pd
import time
import math
sys.path.insert(1,"../Simulation_Pybullet/")
from KinovaGen3Class import KinovaGen3
from Rotations import *

class env_pybullet_kin_gen3:
    "Enviorment of pybullet for the Kinova Gen3 "


class env_pybullet_kin_gen3() :

    def __init__(self,visual=True,Excel_path_Okay = "./Original_Mujoco_Training1_averageconverted.xlsx",\
                    experiment = "Training1",repeats = 3):
        self.visual = visual
        self.experiment = experiment
        self.repeats = repeats

        if (self.visual == True):
            p.connect(p.GUI)
            print("hola")
        else:
            p.connect()
        p.setGravity(0.0, 0.0, -9.81)
        self.robot = KinovaGen3(robot_urdf="../Simulation_Pybullet/models/urdf/JACO3_URDF_V11.urdf")

        self.robot.save_database = False

        #self.init_state_id = p.saveState()



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
        #print("mass")
        #print(self.mass)
        self.mass=list( list(self.mass.reshape(1,7)).pop() )
        #print(self.mass)
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


        self.observation_space = 1
        self.parameters_to_modify = ["mass"]
        self.action_space = len(self.parameters_to_modify)*self.robot.number_robot_control_joints

        self.df_Okay = self.create_df_from_Excel(Excel_path_Okay)

        #print(self.link_name)
        #print(self.joint_id)
        #print(self.parent_link_id)
        #print(self.damping)
    def create_df_from_Excel(self,Excel_path):
        df = pd.read_excel(Excel_path)
        df = df.iloc[:,1:]
        return df

    def action_original(self):
        action_original = []
        for parameter in self.parameters_to_modify:
            action_original= action_original + list(self.original_parameters_df[parameter])
        return action_original

    def action_modified(self):
        action_original = []
        for parameter in self.parameters_to_modify:
            action_original= action_original + list(self.modified_parameters_df[parameter])
        return action_original

    def update_parameters_to_modify(self,parameters_to_modify):
        for parameter in parameters_to_modify:
            if( parameter in list(self.original_parameters_df.columns) ):
                print(parameter+ " okey")
            else:
                    print("One or more parameters selected are not in the possible ones to modify")
                    print("Possibles to modify")
                    print(list(self.original_parameters_df.columns))
                    return
        self.parameters_to_modify = parameters_to_modify
        self.action_space = len(self.parameters_to_modify)*self.robot.number_robot_control_joints



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
        #print("Hola")
        #print(list(action))
        #time.sleep(10)
        #print( len(list(action)) )
        #print(len(self.parameters_to_modify)*self.robot.number_robot_control_joints)
        if( len(list(action)) != len(self.parameters_to_modify)*self.robot.number_robot_control_joints):
            print("The action has not the right length , with the parameters chossen to modify")
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
        parameters_values = list(self.modified_parameters_df["mass"]) +\
                            list(self.modified_parameters_df["damping"])+inertia
        #print(self.modified_parameters_df)
        #print(parameters_values)

        #Start allways in the same state
        #p.restoreState(stateId=self.init_state_id)
        """
        """
        self.robot.modify_robot_pybullet(self.robot.robot_control_joints,\
                                        ["mass","damping","inertia"],\
                                        parameters_values)

        #time.sleep(10)
        self.robot = self.Do_Experiment (repeats=None,experiment=None,kp_list=list(self.modified_parameters_df["kp"]),\
                                                            ki_list=list(self.modified_parameters_df["ki"]),\
                                                            kd_list=list(self.modified_parameters_df["kd"]),\
                                                            max_vel_list=list(self.modified_parameters_df["max_vel"]),\
                                                            force_per_one_list=list(self.modified_parameters_df["force_x_one"]))
        df_test = self.Do_Average_experiments()

        #Erase data from class robot to use it again
        self.robot.database_name_old = None
        self.robot.database_list = []
        self.database_name = "Database"

        self.df_avg = df_test.copy(deep=True)
        reward = self.Compute_Reward(df_test)
        return  reward

    def Do_Experiment(self,repeats=None,experiment=None,robot=None,max_vel_list=[30],force_per_one_list=[1],joint = 1,kp_list=[0.1],ki_list=[0.0],kd_list=[0.0]):

        if(robot == None):
            robot = self.robot
        if(repeats == None):
            repeats = self.repeats
        if(experiment == None):
            experiment = self.experiment

        if (len(max_vel_list) == 1):
            max_vel_list = max_vel_list * robot.number_robot_control_joints
        if (len(force_per_one_list) == 1):
            force_per_one_list = force_per_one_list * robot.number_robot_control_joints

        if (len(kp_list) == 1):
            kp_list = kp_list * robot.number_robot_control_joints
        if (len(ki_list) == 1):
            ki_list = ki_list * robot.number_robot_control_joints
        if (len(kd_list) == 1):
            kd_list = kd_list * robot.number_robot_control_joints

        #print(repeats)
        for iteration in range(repeats):
            # Initialization
            counter = simSteps(experiment,robot.time_step) # detemine time

            #create PIDs
            PID_List = []
            for i in range(robot.number_robot_control_joints ):
                #print(str(max_vel_list[i])+" "+str(kp_list[i])+" "+ str(ki_list[i]) +" "+ str(kd_list[i]))
                PID_List.append( PID(max_velocity=max_vel_list[i],kp=kp_list[i],ki=ki_list[i],kd=kd_list[i]) )
            robot.database_name = "Data_"+str(iteration)
            #print("Values of max vel and kps")
            #time.sleep(10)


            #Move to joint 0
            angles_zero = [0.0]*len(robot.robot_control_joints)
            #print(angles_zero)
            robot.save_database = False
            robot.move_joints(joint_param_value = angles_zero, wait=True)

            #Start saving data every time step
            robot.save_database = True

            #time.sleep(10)
            #Execute experiment during the especified time

            for simStep in range(counter):

                #Every step compute the pid
                PID_List = set_target_thetas(counter, PID_List,experiment,"PyBullet",simStep,joint)

                #Every 0.05 seconds apply the control
                control_steps = int(0.05/robot.time_step)
                if simStep % control_steps == 0:
                    #time.sleep(10)

                    current_angles=robot.get_actual_control_joints_angle()
                    velocity_angles= []

                    #Compute the velocity
                    for i in range( robot.number_robot_control_joints ):
                        #print(i)
                        #print(PID_List[i].get_target_theta())
                        #print(current_angles)
                        velocity_angles.append( PID_List[i].get_velocity(math.degrees(current_angles[i])) /57.32484076 )

                    #Apply the controls
                    robot.move_joints_control_vel( joint_param_value = velocity_angles ,wait = False , desired_force_per_one_list=force_per_one_list)

                robot.step_simulation()


        #Change one lasttime the name and simulate to append it to the database list
        robot.database_name = "dummy"
        robot.step_simulation()
        robot.save_database = False

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

            #print(data.name)
            #print("\n")

        #compute the average, convert all the data to numpy to make it easier
        joint_angles_array = np.array(angles_list)
        #print(joint_angles_array.shape)

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

    def Compute_Reward(self,df_test,df_Okay=None):
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

    def save_parameters(self,Excel_path = "./Parameters_saved.xlsx"):
        self.modified_parameters_df.to_excel(Excel_path)

if (__name__=="__main__"):
    env = env_pybullet_kin_gen3()

    print(env.modified_parameters_df)

    #Test Step elements
    env.robot.visual_inspection = True
    #Modify robot

    aux = np.vstack(( list(env.modified_parameters_df["Ixx"]),\
                    list(env.modified_parameters_df["Iyy"]),
                    list(env.modified_parameters_df["Izz"]) ))
    aux = aux.T

    aux = aux.reshape(1,aux.size)

    inertia = list(list(aux).pop())
    param_values = list(env.modified_parameters_df["mass"]) + list(env.modified_parameters_df["damping"])

    #env.robot.modify_robot_pybullet(env.robot.robot_control_joints,\
                                    #["mass","damping"],\
                                    #param_values\
                                    # )
    time.sleep(10)
    env.robot = env.Do_Experiment (repeats=env.repeats,experiment=None,kp_list=list(env.modified_parameters_df["kp"]),\
                                                        ki_list=list(env.modified_parameters_df["ki"]),\
                                                        kd_list=list(env.modified_parameters_df["kd"]),\
                                                        max_vel_list=list(env.modified_parameters_df["max_vel"]),\
                                                        force_per_one_list=list(env.modified_parameters_df["force_x_one"]))
    df_test = env.Do_Average_experiments(robot=env.robot)

    print(env.df_Okay)
    print(df_test)
    reward = env.Compute_Reward(df_test)
    print(reward)

    #action = [ 1.54286749,  1.19724518,  0.88643233,  1.11632697, -0.29453078, -0.23184369,\
        #-0.06842833, -1.24305685, -1.08163017,  0.20711672,  0.97938179,  0.35433912,\
        #0.89111733, -0.5039466 ]
    #env.step(action)
    while(True):
        time.sleep(0.01)

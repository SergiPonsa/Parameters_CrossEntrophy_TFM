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
                    Excel_path_Okay_tcp = "./positions_from_joints_Mujoco.xlsx",experiment = "Training1",\
                    orient_multiply = 0.009*0.5,repeats = 1, no_zeros = False,time_step=0.05,home_angles=[0, 0.392, 0.0, 1.962, 0.0, 0.78, 0.0]):
        self.visual = visual
        self.experiment = experiment
        self.repeats = repeats
        self.no_zeros = no_zeros
        self.orient_multiply = orient_multiply
        self.home_angles = home_angles

        if (self.visual == True):
            p.connect(p.GUI)
            print("hola")
        else:
            p.connect()
        p.setGravity(0.0, 0.0, -9.81)
        self.robot = KinovaGen3(robot_urdf="../Simulation_Pybullet/models/urdf/JACO3_URDF_V11.urdf",time_step = time_step,home_angles=home_angles)

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
        #to apply the per one
        if(self.no_zeros==True):
            self.no_zero_actions()

        self.action_space = len(self.parameters_to_modify)*self.robot.number_robot_control_joints

        self.df_Okay = self.create_df_from_Excel(Excel_path_Okay)
        self.df_Okay_tcp = self.create_df_from_Excel(Excel_path_Okay_tcp)

        #print(self.link_name)
        #print(self.joint_id)
        #print(self.parent_link_id)
        #print(self.damping)

    def no_zero_actions(self):
        #List and numpy arrays by default are pointers if you don't use copy()
        #For this reason this works
        for element in self.parameters_to_modify:
            joints_parameter = self.mass if element == "mass"\
                            else self.Ixx if element == "Ixx"\
                            else self.Iyy if element == "Iyy"\
                            else self.Izz if element == "Izz"\
                            else self.damping if element == "damping"\
                            else self.kp if element =="kp"\
                            else self.ki if element=="ki"\
                            else self.kd if element =="kd"\
                            else self.max_vel if element=="max_vel"\
                            else self.force_x_one if element =="force_x_one"\
                            else "error"
            value = 0.1 if element =="kp"\
                    else 0.05 if element =="ki"\
                    else 0.05 if element =="kd"\
                    else 0.05 if element =="damping"\
                    else 0.1
            for i in range(self.robot.number_robot_control_joints):
                joints_parameter[i] = value if joints_parameter[i]==0 else joints_parameter[i]

            self.original_parameters_df[element]=joints_parameter

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
        action_modified = []
        for parameter in self.parameters_to_modify:
            action_modified= action_modified + list(self.modified_parameters_df[parameter])
        return action_modified

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
        if(self.no_zeros == True):
            self.no_zero_actions()



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
        if (self.repeats == 1):
            data = self.robot.database_list.pop()

            #create data frame
            df_test = pd.DataFrame({})
            joint_angles = np.array(data.joints_angles_rad)
            for i in range(joint_angles.shape[1]):
                column_name = "joint"+str(i)

                df_test[column_name] = joint_angles[:,i]
            df_test.index = data.time

        else:
            df_test = self.Do_Average_experiments()

        #Erase data from class robot to use it again
        self.robot.database_name_old = None
        self.robot.database_list = []
        self.database_name = "Database"

        self.df_avg = df_test.copy(deep=True)
        reward = self.Compute_Reward(df_test)
        return  reward

    def step_tcp(self,action):
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
        if (self.repeats == 1):
            data = self.robot.database_list.pop()

            #create data frame
            df_test = pd.DataFrame({})
            tcp_position_np = np.array(data.tcp_position)
            for i in range(tcp_position_np.shape[1]):
                column_name = "pos"+str(i)

                df_test[column_name] = tcp_position_np[:,i]

            tcp_orientation_e = np.array(data.tcp_orientation_e)
            for i in range(tcp_orientation_e.shape[1]):
                column_name = "ori"+str(i)

                df_test[column_name] = tcp_orientation_e[:,i]

            df_test.index = data.time

        else:
            df_test = self.Do_Average_experiments_tcp()

        #Erase data from class robot to use it again
        self.robot.database_name_old = None
        self.robot.database_list = []
        self.database_name = "Database"

        self.df_avg = df_test.copy(deep=True)
        reward = self.Compute_Reward_tcp(df_test)
        return  reward

    def step_tcp_euclidian(self,action):
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
        if (self.repeats == 1):
            data = self.robot.database_list.pop()

            #create data frame
            df_test = pd.DataFrame({})
            tcp_position_np = np.array(data.tcp_position)
            for i in range(tcp_position_np.shape[1]):
                column_name = "pos"+str(i)

                df_test[column_name] = tcp_position_np[:,i]

            tcp_orientation_e = np.array(data.tcp_orientation_e)
            for i in range(tcp_orientation_e.shape[1]):
                column_name = "ori"+str(i)

                df_test[column_name] = tcp_orientation_e[:,i]

            df_test.index = data.time

        else:
            df_test = self.Do_Average_experiments_tcp()

        #Erase data from class robot to use it again
        self.robot.database_name_old = None
        self.robot.database_list = []
        self.database_name = "Database"

        self.df_avg = df_test.copy(deep=True)
        reward = self.Compute_Reward_tcp_Euclidian(df_test)
        return  reward

    def step_tcp_rishabh(self,action):
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
        self.robot = self.Do_Experiment_from_Excel_Data (repeats=None,experiment=None,kp_list=list(self.modified_parameters_df["kp"]),\
                                                            ki_list=list(self.modified_parameters_df["ki"]),\
                                                            kd_list=list(self.modified_parameters_df["kd"]),\
                                                            max_vel_list=list(self.modified_parameters_df["max_vel"]),\
                                                    force_per_one_list=list(self.modified_parameters_df["force_x_one"]))
        """
        if (self.repeats == 1):
            data = self.robot.database_list.pop()

            #create data frame
            df_test = pd.DataFrame({})
            tcp_position_np = np.array(data.tcp_position)
            for i in range(tcp_position_np.shape[1]):
                column_name = "pos"+str(i)

                df_test[column_name] = tcp_position_np[:,i]

            tcp_orientation_e = np.array(data.tcp_orientation_e)
            for i in range(tcp_orientation_e.shape[1]):
                column_name = "ori"+str(i)

                df_test[column_name] = tcp_orientation_e[:,i]

            df_test.index = data.time

        else:
            df_test = self.Do_Average_experiments_tcp()
        """
        df_test = self.Do_Average_experiments_tcp()

        #Erase data from class robot to use it again
        self.robot.database_name_old = None
        self.robot.database_list = []
        self.database_name = "Database"

        self.df_avg = df_test.copy(deep=True)
        reward = self.Compute_Reward_tcp_Euclidian(df_test)
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
                    p.stepSimulation()
                robot.step_simulation()


        #Change one lasttime the name and simulate to append it to the database list
        robot.database_name = "dummy"
        robot.step_simulation()
        robot.save_database = False

        return robot

    def Do_Experiment_from_Excel_Data(self,repeats=None,experiment=None,robot=None,max_vel_list=[30],force_per_one_list=[1],joint = 1,kp_list=[0.1],ki_list=[0.0],kd_list=[0.0],\
                                        Excel_Data_List=["Joint_Trajectori_19converted.xlsx",\
                                        "Joint_Trajectori_39converted.xlsx",\
                                        "Joint_Trajectori_59converted.xlsx"],TCP=False,RPY=False,change_joints_time = 0.05,Experiment_time=2):

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
        #print("I Extract data")
        #print(Excel_Data_List)
        for iteration in range(len(Excel_Data_List)):
            # Initialization
            counter = int(Experiment_time/robot.time_step) # detemine time
            #Get joints data
            all_joints = self.Get_joints_from_Excel(Excel_Data_List[iteration],robot = robot,RPY=RPY,TCP=TCP)
            #print("I Exit")
            #print(str(all_joints)+"\n")

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
            robot.move_joints(joint_param_value = self.home_angles, wait=True)

            #Start saving data every time step
            robot.save_database = True

            #time.sleep(10)
            #Execute experiment during the especified time

            #Every x seconds apply the control
            control_steps = int(change_joints_time/robot.time_step)
            #print("control_steps"+str(control_steps))
            #print("counter"+str(counter))
            control_steps = 1 if control_steps == 0 else control_steps

            for simStep in range(counter):
                #print("simStep "+str(simStep))
                #time.sleep(10.0)
                if simStep % control_steps == 0:
                    #print("i apply control")
                    current_angles=robot.get_actual_control_joints_angle()
                    velocity_angles= []
                    joints = all_joints.pop(0)

                    #Compute the velocity and set thetas
                    for i in range( robot.number_robot_control_joints ):
                        #print(i)
                        #print(PID_List[i].get_target_theta())
                        #print(current_angles)
                        PID_List[i].set_target_theta(joints[i],degrees=False)
                        velocity_angles.append( PID_List[i].get_velocity(math.degrees(current_angles[i])) /57.32484076 )
                    #print(str(velocity_angles)+"\n")
                    #time.sleep(100)
                    #Apply the controls
                    robot.move_joints_control_vel( joint_param_value = velocity_angles ,wait = False , desired_force_per_one_list=force_per_one_list)

                robot.step_simulation()


        #Change one lasttime the name and simulate to append it to the database list
        robot.database_name = "dummy"
        robot.step_simulation()
        robot.save_database = False

        return robot

    def Get_joints_from_Excel(self,Excel_Data,robot=None,TCP=True,RPY=False,max_iteration = 10**4):
        if(robot == None):
            robot = self.robot
        #Get the data from the Excel_Data
        Dataframe_Data = pd.read_excel(Excel_Data)
        Datanumpy_Data = Dataframe_Data.to_numpy()
        #Substract the first row which it's the index
        Datanumpy_Data = Datanumpy_Data[:,1:]
        if(TCP == True):
            #I consider the data come like x,y,z,rx,ry,rz
            joints = []
            for i in range(Datanumpy_Data.shape[0]):
                #print(i)
                pose_xyz = Datanumpy_Data[i,:3]
                if(RPY==True):
                    pose_rpy = Datanumpy_Data[i,3:]

                    pose_quat = p.getQuaternionFromEuler(pose_rpy)
                else:
                    pose_quat = Datanumpy_Data[i,3:]

                inv_result = p.calculateInverseKinematics(robot.robot_id, robot.last_robot_joint_index, pose_xyz, pose_quat,
                                                      maxNumIterations = max_iteration)
                joints.append(list(inv_result))
        else:
            joints = list(Datanumpy_Data[:,:])

        return joints

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

    def Do_Average_experiments_tcp(self,robot = None):
        if(robot == None):
            robot = self.robot

        dataframe_list = []
        tcp_position_list = []
        tcp_orientation_e_list = []

        #Go through all the database classes created during the experiment
        for data in robot.database_list:

            #pass the data to a list to do the average
            tcp_position_list.append(data.tcp_position)
            tcp_orientation_e_list.append(data.tcp_orientation_e)

            #print(data.name)
            #print("\n")

        #compute the average, convert all the data to numpy to make it easier
        #print("length tcp position list")
        #print(len(tcp_position_list))
        tcp_position_array = np.array(tcp_position_list)
        #print(tcp_position_array.shape)
        #dimensions of the numpy
        [experiments,steps,tcp_position] = tcp_position_array.shape
        #compute the average, convert all the data to numpy to make it easier
        tcp_orientation_e_array = np.array(tcp_orientation_e_list)
        #print(tcp_orientation_e_array.shape)
        #dimensions of the numpy
        [experiments,steps,tcp_orientation_e] = tcp_orientation_e_array.shape


        average_pos_steps = []
        average_ori_steps = []
        for stp in range(steps):
            average_pos = []
            average_ori = []
            for j in range(tcp_position):
                average_value = np.average(tcp_position_array[:,stp,j])
                average_pos.append(average_value)
                average_value = np.average(tcp_orientation_e_array[:,stp,j])
                average_ori.append(average_value)
            average_pos_steps.append(average_pos)
            average_ori_steps.append(average_ori)

        #create the average data frame
        avg_df = pd.DataFrame({})

        tcp_pos_average = np.array(average_pos_steps)
        tcp_ori_average = np.array(average_ori_steps)

        avg_df["pos x"] = tcp_pos_average [:,0]
        avg_df["pos y"] = tcp_pos_average [:,1]
        avg_df["pos z"] = tcp_pos_average [:,2]

        avg_df["ori x"] = tcp_ori_average [:,0]
        avg_df["ori y"] = tcp_ori_average [:,1]
        avg_df["ori z"] = tcp_ori_average [:,2]

        avg_df.index = data.time

        return avg_df


    def Compute_Reward(self,df_test,df_Okay=None):
        if(df_Okay == None):
            df_Okay = self.df_Okay

        np_Substract = df_test.to_numpy()
        np_ToSubstract = df_Okay.to_numpy()

        if (np_Substract.shape != np_ToSubstract.shape):

            print("Data samples have diferent shapes correct it")

        np_result = np_ToSubstract[:,:] - np_Substract[:,:]
        np_result = np.absolute(np_result) *-1
        reward = np_result.sum()

        return reward

    def Compute_Reward_tcp(self,df_test,df_Okay=None):
        if(df_Okay == None):
            df_Okay = self.df_Okay_tcp

        np_Substract = df_test.to_numpy()
        np_ToSubstract = df_Okay.to_numpy()

        if (np_Substract.shape != np_ToSubstract.shape):

            print("Data samples have diferent shapes correct it")

        np_result = np_ToSubstract[:,:] - np_Substract[:,:]
        np_result[:,3:] = np_result[:,3:] * self.orient_multiply
        np_result = np.absolute(np_result) *-1
        reward = np_result.sum()

        return reward

    def Compute_Reward_tcp_Euclidian(self,df_test,df_Okay=None):
        if(df_Okay == None):
            df_Okay = self.df_Okay_tcp

        np_test = df_test.to_numpy()
        np_Okay = df_Okay.to_numpy()

        #if (np_Okay.shape != np_test.shape):

            #print("Data samples have diferent shapes correct it, if the okay it's in quaternion it will be converted to euler")

        #TCP euclidian distance, dataframes doesn't have time
        euc_distbytime_okay = np_Okay[:-1,0:3]-np_Okay[1:,0:3]
        euc_distbytime_okay = np.multiply(euc_distbytime_okay[:,0],euc_distbytime_okay[:,0]) +\
                        np.multiply(euc_distbytime_okay[:,1],euc_distbytime_okay[:,1]) +\
                        np.multiply(euc_distbytime_okay[:,2],euc_distbytime_okay[:,2])
        euc_distbytime_okay = np.sqrt(euc_distbytime_okay)


        euc_distbytime_test = np_test[:-1,0:3]-np_test[1:,0:3]
        euc_distbytime_test = np.multiply(euc_distbytime_test[:,0],euc_distbytime_test[:,0])+\
                        np.multiply(euc_distbytime_test[:,1],euc_distbytime_test[:,1])+\
                        np.multiply(euc_distbytime_test[:,2],euc_distbytime_test[:,2])

        euc_distbytime_test = np.sqrt(euc_distbytime_test)

        euc_distbytime_btw = np.absolute(euc_distbytime_okay - euc_distbytime_test)
        euc_distbytime_btw = euc_distbytime_btw.sum()/euc_distbytime_btw.shape[0]

        euc_endsdist_btw = np_Okay[:,0:3]-np_test[:,0:3]
        euc_endsdist_btw = np.multiply(euc_endsdist_btw[:,0],euc_endsdist_btw[:,0])+\
                        np.multiply(euc_endsdist_btw[:,1],euc_endsdist_btw[:,1])+\
                        np.multiply(euc_endsdist_btw[:,2],euc_endsdist_btw[:,2])
        euc_endsdist_btw = np.sqrt(euc_endsdist_btw)
        euc_endsdist_btw = euc_endsdist_btw.sum()/euc_endsdist_btw.shape[0]

        if(np_Okay.shape[1]==6):
            np_ori_Okay = np_Okay[:,3:].copy()
        else:
            np_ori_Okay =[]
            for i in range (np_Okay.shape[0]):
                np_ori_Okay.append(p.getEulerFromQuaternion(np_Okay[i,3:]))
            np_ori_Okay = np.array(np_ori_Okay)
        if(np_test.shape[1]==6):
            np_ori_test = np_test[:,3:].copy()
        else:
            np_ori_test =[]
            for i in range (np_Okay.shape[0]):
                np_ori_test.append(p.getEulerFromQuaternion(np_test[i,3:]))
            np_ori_test = np.array(np_ori_test)
        np_ori = np_ori_Okay - np_ori_test
        np_ori = np_ori * self.orient_multiply
        np_ori = np.absolute(np_ori)
        np_ori = np_ori.sum()/np_ori.shape[0]

        reward = -1*euc_distbytime_btw + -1*euc_endsdist_btw + -1*np_ori

        return reward

    def save_parameters(self,Excel_path = "./Parameters_saved.xlsx"):
        self.modified_parameters_df.to_excel(Excel_path)

if (__name__=="__main__"):
    env = env_pybullet_kin_gen3()

    Reward_opt = 2 #0 reward using joints | 1 reward using tcp | 2 reward using tcp Rishab

    print(env.modified_parameters_df)

    #Test Step elements
    env.robot.visual_inspection = False
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
    if(Reward_opt == 0):
        if(env.repeats ==1):
            data = env.robot.database_list.pop()

            #create data frame
            df_test = pd.DataFrame({})
            joint_angles = np.array(data.joints_angles_rad)
            print(joint_angles)
            print(joint_angles.shape)
            for i in range(joint_angles.shape[1]):
                print(i)
                column_name = "joint"+str(i)

                df_test[column_name] = joint_angles[:,i]
            print(len(data.time))
            print(data.time)
            df_test.index = data.time

        else:
            df_test = env.Do_Average_experiments(robot=env.robot)
    else:
        if(env.repeats ==1):
            data = env.robot.database_list.pop()

            #create data frame
            df_test = pd.DataFrame({})
            tcp_position_np = np.array(data.tcp_position)
            for i in range(tcp_position_np.shape[1]):
                column_name = "pos"+str(i)

                df_test[column_name] = tcp_position_np[:,i]

            tcp_orientation_e = np.array(data.tcp_orientation_e)
            for i in range(tcp_orientation_e.shape[1]):
                column_name = "ori"+str(i)

                df_test[column_name] = tcp_orientation_e[:,i]

            df_test.index = data.time

        else:
            df_test = env.Do_Average_experiments_tcp(robot=env.robot)



    if(Reward_opt == 0):
        print(env.df_Okay)
        print(df_test)
        df_test.to_excel("Test_enverionment.xlsx")
        reward = env.Compute_Reward(df_test)
    else:
        print(env.df_Okay_tcp)
        print(df_test)
        df_test.to_excel("Test_enverionment_tcp.xlsx")

        if(Reward_opt == 1):
            reward = env.Compute_Reward_tcp(df_test)
        else:
            reward = env.Compute_Reward_tcp_Euclidian(df_test)
    print(reward)



    #action = [ 1.54286749,  1.19724518,  0.88643233,  1.11632697, -0.29453078, -0.23184369,\
        #-0.06842833, -1.24305685, -1.08163017,  0.20711672,  0.97938179,  0.35433912,\
        #0.89111733, -0.5039466 ]
    #env.step(action)
    while(True):
        time.sleep(0.01)

import numpy as np
import pandas as pd

def Numpy_file_to_df_actions(numpy_file_name="acs.npy",time_step=10**-4,episode=19,multiplier=0.05):
    numpy_data = np.load(numpy_file_name)
    dataframe = pd.DataFrame({})
    dataframe["offsetx"] =list(numpy_data[episode,:,0]*multiplier)
    dataframe["offsety"] =list(numpy_data[episode,:,1]*multiplier)
    dataframe["offsetz"] =list(numpy_data[episode,:,2]*multiplier)
    dataframe["gripper"] =list(numpy_data[episode,:,3]*multiplier)
    time = np.arange(0,time_step*numpy_data.shape[1],time_step)
    dataframe.index=time

    return dataframe

def Numpy_file_to_df_acumulative(numpy_file_name="acs.npy",time_step=10**-4,episode=19,multiplier=0.05):
    numpy_data = np.load(numpy_file_name)
    numpy_data = numpy_data*multiplier
    numpy_data_acum = numpy_data[episode,:,:].copy()
    print(numpy_data_acum.shape)
    numpy_data_acum = np.cumsum(numpy_data_acum,axis=0)
    print(numpy_data_acum.shape)
    numpy_data_acum = numpy_data_acum.reshape( (numpy_data.shape[1],numpy_data.shape[2]) )
    print(numpy_data_acum.shape)
    dataframe = pd.DataFrame({})
    dataframe["offsetx-acum"] =list(numpy_data_acum[:,0])
    dataframe["offsety-acum"] =list(numpy_data_acum[:,1])
    dataframe["offsetz-acum"] =list(numpy_data_acum[:,2])
    dataframe["gripper-acum"] =list(numpy_data_acum[:,3])

    time = np.arange(0,time_step*numpy_data.shape[1],time_step)
    dataframe.index=time

    return dataframe

def Numpy_file_to_df_tcp(numpy_file_name="acs.npy",time_step=10**-4,episode=19,multiplier=0.05,init_tcp=[1.0,1.0,1.0,3.14,0.0,0.0]):
    numpy_data = np.load(numpy_file_name)
    numpy_data = numpy_data*multiplier
    numpy_data_acum = numpy_data[episode,:,:].copy()
    print(numpy_data_acum.shape)
    numpy_data_acum = np.cumsum(numpy_data_acum,axis=0)
    print(numpy_data_acum.shape)
    numpy_data_acum = numpy_data_acum.reshape( (numpy_data.shape[1],numpy_data.shape[2]) )
    print(numpy_data_acum.shape)
    dataframe = pd.DataFrame({})

    initx = np.ones([1,numpy_data.shape[1]])*init_tcp[0]
    inity = np.ones([1,numpy_data.shape[1]])*init_tcp[1]
    initz = np.ones([1,numpy_data.shape[1]])*init_tcp[2]
    initrx = np.ones([numpy_data.shape[1],1])*init_tcp[3]
    initry = np.ones([numpy_data.shape[1],1])*init_tcp[4]
    initrz = np.ones([numpy_data.shape[1],1])*init_tcp[5]

    print(numpy_data_acum.shape)
    print(numpy_data_acum[:,0].size)
    print(initx.size)
    print(numpy_data_acum[:,0])
    print( numpy_data_acum[:,0] + initx )

    dataframe["tcp-x"] = list(list(numpy_data_acum[:,0] + initx).pop())
    dataframe["tcp-y"] = list(list(numpy_data_acum[:,1] + inity).pop())
    dataframe["tcp-z"] = list(list(numpy_data_acum[:,2] + initz).pop())
    dataframe["tcp-rx"] = list(initrx)
    dataframe["tcp-ry"] = list(initry)
    dataframe["tcp-rz"] = list(initrz)
    time = np.arange(0,time_step*numpy_data.shape[1],time_step)
    print(time)
    dataframe.index=time

    return dataframe

if __name__=="__main__":
    dataframe = Numpy_file_to_df_tcp()
    dataframe.to_excel("TCP_pos.xlsx")

    dataframe = Numpy_file_to_df_acumulative()
    dataframe.to_excel("TCP_accum.xlsx")

    dataframe = Numpy_file_to_df_actions()
    dataframe.to_excel("TCP_actions.xlsx")

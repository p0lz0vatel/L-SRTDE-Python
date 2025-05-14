import os
import numpy as np
from scipy.io import loadmat
import sys 

np.set_printoptions(precision=25,threshold=sys.maxsize,linewidth=5000)
current_dir = os.getcwd()
folder_path = os.path.join(current_dir)

for ProblemIndex in range(1,25):
    print(ProblemIndex)
    if 1 <= ProblemIndex <= 24:
        filename = f'f{ProblemIndex}.mat'
        GNBG_tmp = loadmat(os.path.join(folder_path, filename))['GNBG']
        MaxEvals = np.array([item[0] for item in GNBG_tmp['MaxEvals'].flatten()])[0, 0]
        AcceptanceThreshold = np.array([item[0] for item in GNBG_tmp['AcceptanceThreshold'].flatten()])[0, 0]
        Dimension = np.array([item[0] for item in GNBG_tmp['Dimension'].flatten()])[0, 0]
        CompNum = np.array([item[0] for item in GNBG_tmp['o'].flatten()])[0, 0]  # Number of components
        MinCoordinate = np.array([item[0] for item in GNBG_tmp['MinCoordinate'].flatten()])[0, 0]
        MaxCoordinate = np.array([item[0] for item in GNBG_tmp['MaxCoordinate'].flatten()])[0, 0]
        CompMinPos = np.array(GNBG_tmp['Component_MinimumPosition'][0, 0])
        CompSigma = np.array(GNBG_tmp['ComponentSigma'][0, 0], dtype=np.float64)
        CompH = np.array(GNBG_tmp['Component_H'][0, 0])
        Mu = np.array(GNBG_tmp['Mu'][0, 0])
        Omega = np.array(GNBG_tmp['Omega'][0, 0])
        Lambda = np.array(GNBG_tmp['lambda'][0, 0])
        RotationMatrix = np.array(GNBG_tmp['RotationMatrix'][0, 0])
        OptimumValue = np.array([item[0] for item in GNBG_tmp['OptimumValue'].flatten()])[0, 0]
        OptimumPosition = np.array(GNBG_tmp['OptimumPosition'][0, 0])
    else:
        raise ValueError('ProblemIndex must be between 1 and 24.')
    contents = "";
    contents += np.array2string(MaxEvals)+"\n"
    contents += np.array2string(AcceptanceThreshold)+"\n"
    contents += np.array2string(Dimension)+"\n"
    contents += np.array2string(CompNum)+"\n"
    contents += np.array2string(MinCoordinate)+"\n"
    contents += np.array2string(MaxCoordinate)+"\n"
    contents += np.array2string(CompMinPos)+"\n"
    contents += np.array2string(CompSigma)+"\n"
    contents += np.array2string(CompH)+"\n"
    contents += np.array2string(Mu)+"\n"
    contents += np.array2string(Omega)+"\n"
    contents += np.array2string(Lambda)+"\n"
    contents += np.array2string(RotationMatrix)+"\n"
    contents += np.array2string(OptimumValue)+"\n"
    contents += np.array2string(OptimumPosition)+"\n"
    contents = contents.replace("[","")
    contents = contents.replace("]","")
    contents = contents.replace("\n\n","\n")
    #print(contents)
    with open(f"f{ProblemIndex}.txt", "w") as text_file:
        text_file.write(contents)

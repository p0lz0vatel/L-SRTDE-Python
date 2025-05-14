import numpy as np

PATH = "Results/"
NFunc = 24
NRuns = 31
AllRes = np.zeros((NFunc,NRuns,1001))
for func in range(NFunc):
    AllRes[func] = np.loadtxt(PATH+f"L-SRTDE_GNBG_F{func+1}_D30.txt")
ResTable = np.zeros((5,NFunc))
str1 = "Func & Absolute error & Required FEs to Acceptance Threshold & Success rate \\\\\n"
for func in range(NFunc):
    str1 += f'F{func+1} & '
    str1 += f'${np.mean(AllRes[func,:,-2]):.6g} \pm {np.std(AllRes[func,:,-2]):.6g}$ & '
    str1 += f'${np.mean(AllRes[func,:,-1]):.6g} \pm {np.std(AllRes[func,:,-1]):.6g}$ & '
    Success = 0
    for j in range(NRuns):        
        Success += (AllRes[func,j,-2] == 0)
    Success /= NRuns
    str1 += f'${Success:.6g}$ & '
    ResTable[0][func] = np.mean(AllRes[func,:,-2])
    ResTable[1][func] = np.std( AllRes[func,:,-2])
    ResTable[2][func] = np.mean(AllRes[func,:,-1])
    ResTable[3][func] = np.std( AllRes[func,:,-1])
    ResTable[4][func] = Success
    str1=str1[:-2]+"\\\\\n"
print(str1)
np.savetxt(PATH+"Results_Table_L-SRTDE_GNBG24.txt",ResTable)

ResTable = np.zeros((5,NFunc))
str1 = "Func & "
for func in range(NFunc):
    str1 += f'F{func+1} & '
str1 = str1[:-2]+"\\\\\n"
str1 += "Absolute error & "
for func in range(NFunc):
    str1 += f'${np.mean(AllRes[func,:,-2]):.6g} \pm {np.std(AllRes[func,:,-2]):.6g}$ & '
str1=str1[:-2]+"\\\\\n"
str1 += "Required FEs to Acceptance Threshold & "
for func in range(NFunc):
    str1 += f'${np.mean(AllRes[func,:,-1]):.6g} \pm {np.std(AllRes[func,:,-1]):.6g}$ & '
str1=str1[:-2]+"\\\\\n"
str1 += "Success rate & "
for func in range(NFunc):
    Success = 0
    for j in range(NRuns):        
        Success += (AllRes[func,j,-2] == 0)
    Success /= NRuns
    str1 += f'{Success:.6g} & '
str1=str1[:-2]+"\\\\\n"
print(str1)

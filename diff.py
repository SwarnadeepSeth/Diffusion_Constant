import numpy as np
import matplotlib.pyplot as plt

cm = np.loadtxt("cm.dat", delimiter=' ', usecols=(1, 2, 3), unpack=True) 
#cm = np.loadtxt("cm_exam.dat", delimiter=' ', usecols=(1, 2, 3), unpack=True) 
#cm = np.loadtxt("cm_example.dat", delimiter=' ', usecols=(1, 2, 3), unpack=True) 

total_traj_length = len(cm[1])

n_simulations = 100
# create n_simulations: 100 independent trajectories by chopping
n_steps = int(total_traj_length/n_simulations)

position = np.zeros((n_simulations, n_steps))
for i in range(n_simulations):
	begin_index = i*n_steps
	
	cnt = 0
	for j in range (begin_index, begin_index+n_steps):
		#position[i, cnt] =  cm[1][j]
		position[i, cnt] = (cm[1][j]-cm[1][begin_index])
		cnt = cnt +  1

msd = np.zeros(n_steps)
t_array=[]
for i in range(n_steps):
    msd[i] = np.mean(position[:, i]**2)
    t_array.append(i)

t_array = np.array(t_array)
print (msd)

# Fit the curve
m,b = np.polyfit(t_array, msd, 1)
print("slope:", round(m,2), "Intercept:", round(b,2))

import statsmodels.api as sm
res = sm.OLS(msd, t_array).fit()
print ("Forced Intercept to Zero. Slope is:", res.params)

f = plt.subplots(figsize=(10, 5))
plt.plot(t_array, msd, '-', lw=1.5, c='blue')
plt.plot(t_array, m*t_array+b, '--', color= 'green', label=r'$slope=%s$' %(round(m,3))) 

plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$\langle Y_{cm}(t).Y_{cm}(t)\rangle $', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend( fontsize = 16, frameon=False)

plt.savefig("msd_diff.pdf")

plt.show()





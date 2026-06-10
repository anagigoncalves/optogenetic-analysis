import numpy as np
import math
import matplotlib.pyplot as plt

# right fast
ff = 4 
sf = 4
time = np.linspace(0,500,500)
fr = 1.3*np.sin(2*math.pi*ff*time*0.001-2)-0.3
fl = np.sin(2*math.pi*sf*time*0.001)
time_db_corr1 = np.linspace(0,200,200)
time_db_corr2 = np.linspace(250,450,200)
time_db_corr3 = np.linspace(450,500,50)
db_corr_ltd1 = np.exp(-(time_db_corr1-150)/1000)*(np.power(np.sin(2*math.pi*(time_db_corr1-150)/1000),20))/ 1.6     #1.2848
db_corr_ltd2 = np.exp(-(time_db_corr1-150)/1000)*(np.power(np.sin(2*math.pi*(time_db_corr1-150)/1000),20))/ 1.6     #1.2848
db_corr_ltd3 = np.exp(-(time_db_corr3-150)/1000)*(np.power(np.sin(2*math.pi*(time_db_corr3-150)/1000),20))/ 1.6     #1.2848
db_corr_ltd = np.hstack((db_corr_ltd2, db_corr_ltd3))
db_corr_ltd = np.hstack((db_corr_ltd, db_corr_ltd2))
db_corr_ltd = np.hstack((db_corr_ltd, db_corr_ltd3))
db_corr = np.exp(-(time-150)/1000)*(np.power(np.sin(2*math.pi*(time-150)/1000),20))/ 1.6     #1.2848
db_corr_ltd = np.hstack((db_corr[300:],db_corr_ltd2[:50]))
db_corr_ltd = np.hstack((db_corr_ltd, db_corr[300:]))
db_corr_ltd = np.hstack((db_corr_ltd, db_corr_ltd2[:50]))


plt.plot(time, db_corr)
plt.show()

ub_corr = -0.3*np.ones(time.shape)

cosp_times = []

plt.plot(db_corr_ltd)
plt.plot(db_corr_ltd1,'k')
plt.plot( db_corr_ltd2,'g')
plt.plot( db_corr_ltd3,'y')
plt.show()

plt.plot(time, fr, 'gray')
plt.plot(time, fl, 'b')
plt.plot(time, fr+db_corr_ltd, 'r')
plt.axvline(x=200, linestyle='--')
plt.axvline(x=400, linestyle='--')

plt.show()



# Complex spike on swing
plt.plot(time, db_corr)
plt.axvline(x=300, linestyle='--')
plt.show()

db_corr_ltd = np.hstack((db_corr_ltd2[:100],db_corr[300:]))
db_corr_ltd = np.hstack((db_corr_ltd,db_corr_ltd2[:200]))
plt.plot(time, fr, 'gray')
plt.plot(time, fl, 'b')
plt.plot(time, fr+db_corr_ltd, 'r')
plt.axvline(x=300, linestyle='--')

plt.show()

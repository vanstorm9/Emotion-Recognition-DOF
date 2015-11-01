import numpy as np
import math
from time import time
from collections import Counter
import matplotlib.pyplot as plt

t0 = time()
t1 = time()
print 'Loading the main matrix. . .'
main = np.load('optical-main-mini.npy')
diff = diff = time() - t1
print 'Loaded main matrix in ', diff, 's of size ', main.shape

t2 = time()

print 'Loading the target vector. . .'
target = np.load('optical-target-mini.npy')
diff = time() - t2
print 'Loaded target in ', diff, 's of size ', target.shape

j = 0
mas = 282
max_val = 375
master = np.array([])


while True:
    print j
    if mas > max_val:
        break
    
    #print '----------------', mas, '----------------' 
    #print 'Getting top 100 optical flow indexes'
    idx_main = (-main[j]).argsort()[:100]
    
    #print idx_main

    
    if j == 0:
        master = idx_main
    else:
        master = np.concatenate((master, idx_main))
    
    
    '''
    j = 0
    for idx_idx in idx_main:

        if j == 0:
            master = np.array([idx_idx , 0 , main[mas,idx_idx]])[None,:]
            
            #master = master[None,:]
        else:
            sub_master = np.array([idx_idx , 0 , main[mas, idx_idx]])[None,:]
            master = np.concatenate((master, sub_master))

        master = master[master[:,2].argsort()]
        
            
        j = j + 1
    print 'Done'
    '''
    

    #print 'Printing out results'
    i = 1
    '''
    for idx in idx_main:
        print i, ': ', idx ,' ', main[mas,idx]
        i = i + 1
    '''
    j = j + 1
    mas = mas + 1

master = master / 15

master = master.tolist()
counter = Counter(master)

top_hundred = [ite for ite, it in Counter(counter).most_common(100)]

print 'Top optical points'
print top_hundred

print ''
print ''
print ''

print 'Frequency list'
z = 0


x = np.array([])
y = np.array([])
for i in top_hundred:
    print i, ' ', counter[i]

    if z == 0:
        x = np.array([i])
        y = np.array([counter[i]])
    else:
        sub_x = np.array([i])
        sub_y = np.array([counter[i]])
        x = np.concatenate((x, sub_x))
        y = np.concatenate((y, sub_y))
    z = z + 1
N = 50

colors = np.random.rand(N)
area = np.pi * (10) # 0 to 15 point radiuses

plt.scatter(x, y, s=area, c=colors, alpha=0.5)

plt.title('Flow Points Distributon Across All Emotions')
plt.xlabel('Flow Point Number', fontsize=18)
plt.ylabel('Frequency of Overlap Movement', fontsize=16)

plt.show()


print 'Done!'

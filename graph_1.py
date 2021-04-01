import numpy as np
import matplotlib.pyplot as plt

plt.xlabel("Accuracy (%)")
plt.ylabel("Parameters (Millions)")
plt.xlim((80, 100))
plt.ylim((30, 0))

# x1=[80, 85, 90]
# y1=[35, 30, 25, 20, 15, 10, 5]
vgg_x = [83.0]
vgg_y = [20.37]

Res_x = [90.6]
Res_y = [24.97]

D_m_x = [93.7]
D_m_y = [3.88]

I_x = [91.3]
I_y = [22.98]

CO_x = [93.3]
Co_y = [11.75]

Our_x = [95.3]
Our_y = [3.88]

l1=plt.scatter(vgg_x,vgg_y,label='VGG-19', marker='s')
l2=plt.scatter(Res_x,Res_y,label='ResNet-50', marker='>')
l3=plt.scatter(D_m_x,D_m_y,label='Dilated MobileNet', marker='o', c='b')
l1=plt.scatter(I_x,I_y,label='InceptionV3', marker='8')
l2=plt.scatter(CO_x,Co_y,label='COVID-NET', marker='p')
l3=plt.scatter(Our_x,Our_y,label='RAM-Net (Ours)', marker='*', c='r')



# plt.plot(x1,y1,'ro-',x2,y2,'g+-',x3,y3,'b^-')
plt.title('The Lasers in Three Conditions')

plt.grid(True)

plt.legend()
plt.savefig('test.png')
plt.show()
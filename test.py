import numpy as np
import matplotlib.pyplot as plt

# Make a fake dataset
height = [61.57, 69.72, 71.94]
bars = ('Sequence Features', 'Profile Features', 'Sequence and Profile Features')
y_pos = np.arange(len(bars))
x_pos = np.arange(len(height))
label = ['61.57%', '69.72%', '71.94%']
plt.bar(y_pos, height, color=['black', 'red', 'blue'])
plt.xticks(y_pos, bars, fontsize=8)
plt.xlabel('Different Input Features')
plt.ylabel('Q8 Accuracy (%)')

for i in range(3):
    plt.text(x = y_pos[i]-0.1, y = height[i]+0.5, s = label[i], size = 12)

plt.show()

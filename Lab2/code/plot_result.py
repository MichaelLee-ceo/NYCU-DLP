import matplotlib.pyplot as plt
import numpy as np

file = open('./src/result.txt', 'r')
lines = file.readlines()

score = []
for line in lines:
    score.append(float(line))
file.close()

total_epoch = len(lines) * 1000
print('Total epoch: {}'.format(total_epoch))

x = np.arange(0, total_epoch, 1000)

plt.figure(figsize=(10, 6))
plt.title('Training score')
plt.plot(x, score)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.savefig('result.png')
plt.show()
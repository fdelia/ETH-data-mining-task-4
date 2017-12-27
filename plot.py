# use './run.sh &> output.csv' to produce output

import re
import matplotlib.pyplot as plt

f = open("output.csv", "r")

x = []
y = []
for line in f:
    try:
        r = re.findall("\d+\.\d+", line)
        num = r[0]
    except ValueError:
        print("Error while parsing following line to float:")
        print(line)
        break

    if len(x) == len(y):
        x.append(num)
    else:
        y.append(num)

if len(x) > len(y):
    x = x[:-1]
plt.plot(x, y)

plt.xlabel('parameter')
plt.ylabel('score')
plt.grid(True)
plt.savefig("output.png")
plt.waitforbuttonpress()
plt.show()

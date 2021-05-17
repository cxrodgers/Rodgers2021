## Simple test script, designed to break if the imports or
## plotting aren't configured correctly

import MCwatch.behavior
import matplotlib
import matplotlib.pyplot as plt

f, ax = plt.subplots()
ax.plot([1,2,3])
plt.show()
print("test test test")


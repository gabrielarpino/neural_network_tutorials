import matplotlib.pyplot as plt
import numpy as np

#import plotly.plotly as py
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

prob_list = []
for i in xrange(10000):
	prob_list.append(np.random.binomial(50,0.5))


plt.hist(prob_list)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")

fig = plt.gcf()
plt.draw()
plt.show()

	
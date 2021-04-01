import matplotlib.pyplot as plt

year = [2014, 2015, 2016, 2017, 2018, 2019]
tutorial_count = [39, 117, 111, 110, 67, 29]

year = [2014, 2015, 2016, 2017, 2018, 2019]
tutorial_count1 = [80, 114, 10, 100, 85, 80]

plt.plot(year, tutorial_count)
plt.plot(year, tutorial_count1)
plt.xlabel('Year')
plt.ylabel('Number of futurestud.io Tutorials')

plt.show()
# Take trained model and display the difference between test points and predictions

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def get_map():
    m = Basemap(projection='mill',
                llcrnrlat=25,
                llcrnrlon=-130,
                urcrnrlat=50,
                urcrnrlon=-60,
                resolution='l')

    m.drawcoastlines(linewidth=0.5,color='gray')
    m.drawcountries(linewidth=0.5,color='gray')
    m.drawstates(color='gray')

    return m

m = get_map()
nylat,nylon = 40.7,-74

xpt,ypt = m(nylat,nylon)
m.plot(xpt,ypt,'co')
plt.show()
import math,sys,os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#need to import contours
#from matplotlib.colors import Normalize
import numpy as np
from scipy import interpolate as interp

try:
    sys.path.append('libs/') #add the libs to the system path so we can import them
    #import matrix_solver as ms #add custom modules to this line
    import setup_2d_PDE as pd
except ModuleNotFoundError:
    print('Please only run this from the PRH_Project2 directory.')
    raise ModuleNotFoundError
    sys.exit()

#be aware that grabbing from the config file and setting up module-level variables
#happens inside setup_2d_PDE

#first, write the maps for the two structured cases, just to prove that's programmatic instead of bespoke
#also provides true flexibility, since we don't want to read problems.cfg and then ignore it    
pd.writeStructTriMap(pd.NX, pd.NY, pd.MIN_X, pd.MAX_X, pd.MIN_Y, pd.MAX_Y, pd.DX, pd.DY, 'data/')
pd.writeStructRectMap(pd.NX, pd.NY, pd.MIN_X, pd.MAX_X, pd.MIN_Y, pd.MAX_Y, pd.DX, pd.DY, 'data/')

#then load those same maps. The above two lines are a vanity call and a safeguard, nothing more.
structTriElemMap, structTriCoords = pd.loadElems('data/tri_struct.data') #get the maps for the structured triangle case
structRectElemMap, structRectCoords = pd.loadElems('data/rect_struct.data') #ditto for structured rectangles
unstructTriElemMap, unstructTriCoords = pd.loadElems('data/tri_unstruct.data') #ditto for unstructured triangles

structTriStiff, structTriF = pd.buildStiffness(structTriElemMap, structTriCoords, pd.poissonTriElement) #create the global matrix equation
structRectStiff, structRectF = pd.buildStiffness(structRectElemMap, structRectCoords, pd.poissonRectElement)
unstructTriStiff, unstructTriF = pd.buildStiffness(unstructTriElemMap, unstructTriCoords, pd.poissonTriElement)

structTriKnownNodes, structTriStiff, structTriQ = pd.applyBounds(structTriStiff, structTriF, structTriCoords) #apply boundary conditions to the global matrix equation
structRectKnownNodes, structRectStiff, structRectQ = pd.applyBounds(structRectStiff, structRectF, structRectCoords)
unstructTriKnownNodes, unstructTriStiff, unstructTriQ = pd.applyBounds(unstructTriStiff, unstructTriF, unstructTriCoords)

structTriT = pd.solveU(structTriKnownNodes, structTriStiff, structTriQ) #find the solution, including re-inserting the appropriate boundary node values
structRectT = pd.solveU(structRectKnownNodes, structRectStiff, structRectQ)
unstructTriT = pd.solveU(unstructTriKnownNodes, unstructTriStiff, unstructTriQ)


#our solutions scaled against T0
scaledStructTriT = structTriT/pd.T0 
scaledStructRectT = structRectT/pd.T0
scaledUnstructTriT = unstructTriT/pd.T0

#now write our comparison to file

with open('results/structured_comparison_table for (dx,dy) = ({0},{1}).data'.format(pd.DX,pd.DY), 'w') as file:
    file.write('Table of comparison: structured rectangle, structured triangle, analytic\n')
    file.write('Contains entries for each method in form T/T0\n')
    file.write('\n')
    file.write('{0:<{width}} {1:<{width}} {2:<{width}} {3:<{width}}\n'.format('Node','rect', 'tri', 'analytic',width=9))
    for i in range(1, len(structTriCoords)+1):
        x, y = structTriCoords[i]
        rect = scaledStructRectT[i-1,0]
        tri = scaledStructTriT[i-1,0]
        ana = pd.analyticTemp(x, y)/pd.T0
        row = '{0:{sign}<9.5g} {1:{sign}<9.5g} {2:{sign}<9.5g} {3:{sign}<9.5g}\n'.format(i, rect, tri, ana, sign=' ')

        file.write(row)
        
with open('results/unstructured_comparison_table.data', 'w') as file:
    file.write('Table of comparison: unstructured triangle vs analytic\n')
    file.write('Contains entries in form T/T0\n')
    file.write('\n')
    file.write('{0:<{width}} {1:<{width}} {2:<{width}}\n'.format('Node','triangle', 'analytic',width=9))
    
    for i in range(1, len(unstructTriCoords)+1):
        x, y = unstructTriCoords[i]
        tri = scaledUnstructTriT[i-1,0]
        ana = pd.analyticTemp(x, y)/pd.T0
        row = '{0:{sign}<9.5g} {1:{sign}<9.5g} {2:{sign}<9.5g}\n'.format(i, tri, ana, sign=' ')
  
        file.write(row)
        

#files are written with our comparison data; now it's time to graph.

X = np.linspace(pd.MIN_X, pd.MAX_X, int(1e3//pd.DY)) #1000 times smaller than the node step
Y = np.linspace(pd.MIN_Y, pd.MAX_Y, int(1e3//pd.DY)) #could be a rectangular grid rather than square - but that's fine

#X, Y are our vectors for the X, Y axes

Xfield, Yfield = np.meshgrid(X, Y) #meshgrid for plotting.

#griddata next. scipy.interpolate.griddata takes a set of disconnected points in (x,y) and their values F --> F(x,y)
#and interpolates them against a meshgrid in X, Y - I think.

#in order to do this, we find the X, Y coords of each node in the coordinate map
#and then we find our 'F' in the appropriate u vector solution to our stiffness matrix equation

#also, we want to graph T/T0 to avoid scaling issues.

#structured triangles
structTriPoints = np.array([structTriCoords[i] for i in structTriCoords]) #x and y coords
structTriGrid = interp.griddata(structTriPoints, np.squeeze(np.array(scaledStructTriT)), (Xfield, Yfield)) #uuuugly, but works

#structured rectangles
structRectPoints = np.array([structRectCoords[i] for i in structRectCoords]) #x and y coords
structRectGrid = interp.griddata(structRectPoints, np.squeeze(np.array(scaledStructRectT)), (Xfield, Yfield))

#unstructured triangles
unstructTriPoints = np.array([unstructTriCoords[i] for i in unstructTriCoords]) #x and y coords
unstructTriGrid = interp.griddata(unstructTriPoints, np.squeeze(np.array(scaledUnstructTriT)), (Xfield, Yfield))

#analytical solution - sorta. Still going to use griddata, because that's a lot easier than manually building the grid.

#build the whole analytical matrix by hand in nested lists, then convert to a np type array
anaGrid = np.array([[pd.analyticTemp(i, j)/pd.T0 for i in X]for j in Y])


#plotting setup done, so now it's time to make each plot, label it, and show it

structTriPlot = plt.figure(1)
plt.contourf(X, Y, structTriGrid, origin='lower')
structTriPlot.suptitle('Temperature contours, structured triangles (scaled to T0)')
plt.title('(X,Y) steps = ({0},{1})'.format(pd.DX, pd.DY))
plt.colorbar()

#don't clobber existing files - this can mean a lot of copies
copynum = 0
structTriPath = 'pics/structured_triangles_contour{0}, x,y step ({1}, {2}).png'.format(copynum, pd.DX, pd.DY)
while os.path.isfile(structTriPath):
    copynum +=1
    structTriPath = 'pics/structured_triangles_contour{0}, x,y step ({1}, {2}).png'.format(copynum, pd.DX, pd.DY)
plt.savefig(structTriPath)

unstructTriPlot = plt.figure(2)
plt.contourf(X, Y, unstructTriGrid, origin='lower')
unstructTriPlot.suptitle('Temperature contours, unstructured triangles (scaled to T0)')
plt.colorbar()

unstructTriPath = 'pics/unstructured_triangles_contour.png'
plt.savefig(unstructTriPath)

structRectPlot = plt.figure(3)
plt.contourf(X, Y, structRectGrid, origin='lower')
structRectPlot.suptitle('Temperature contours, structured rectangles (scaled to T0)')
plt.title('(X,Y) steps = ({0},{1})'.format(pd.DX, pd.DY))
plt.colorbar()

copynum = 0
structRectPath = 'pics/structured_rectangles_contour{0}, x,y step ({1}, {2}).png'.format(copynum, pd.DX, pd.DY)
while os.path.isfile(structRectPath):
    copynum +=1
    structRectPath = 'pics/structured_rectangles_contour{0}, x,y step ({1}, {2}).png'.format(copynum, pd.DX, pd.DY)
plt.savefig(structRectPath)

anaPlot = plt.figure(4)
plt.contourf(X, Y, anaGrid, origin='lower')
anaPlot.suptitle('Temperature contours, analytical (scaled to T0)')
plt.colorbar()

anaPath = 'pics/analytical_contour.png'
plt.savefig(anaPath)

plt.show()

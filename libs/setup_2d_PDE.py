import os, sys, math
import numpy as np

#TODO: Put together stuff for a config file. But for now, we can just use the defaults

def loadElems(filename):
    '''Given a filename string, build a dict that describes the node mapping for
each element in the file and gives the x,y positions of each node, indexed by its
global number'''

    with open(filename) as file:
        elemMap = {} #you might ask: why dict? Python is 0 indexed; the problem starts at element 1, and I don't want to get confused. It might be a bad decision.
        coordMap = {}
        mapSection = True #start with mapping

        for line in file:
            line=line.strip().rstrip() #get rid of trailing and leading whitespace
            if line and not line.startswith('#'):

                if 'coord' in line: #we've switched to the coordinates
                    mapSection = False
                    continue #this is just a toggle, so don't process it below

                if mapSection: #global to local map
                    maplist = line.split()
                    
                    #print(maplist)
                    elem = maplist[0]
                    nodes = maplist[1:]
                    elemMap[int(elem)] = [int(n) for n in nodes]

                else: #coordinates
                    #file contains a lot of crap. We only care about the first three entries, which are node number, x coord and y coord
                    coordList = line.split()
                    elem = coordList[0]
                    coords = coordList[1:3]
                    coordMap[int(elem)] = [float(c) for c in coords] 
                

    return(elemMap, coordMap)

class UnevenStructure(Exception): #use for writeStrucSquareMap when the specified values don't add up properly
    pass

def writeStructRectMap(nx, ny, xmin, xmax, ymin, ymax, dx, dy, path):
    '''Given the number of nodes in the x and y directions, as well as the minimum
and maximum range values in x and y, and the jump steps in x and y, write a structured rectangular map to the given
filepath named rect_struct.data'''

    nelem = (nx-1)*(ny-1) #total number of elements
    if (xmax-xmin)%((nx-1)*dx) or (ymax-ymin)%((ny-1)*dy): #This says that if the interval covered by X or Y doesn't fit an even multiple of the number of jumps in that direction, the domain doesn't fit the jumps
        #ny or ny - 1 because we need to consider the nodes at xmin and ymin as well. Effectively nx-1 is number of x elements
        raise UnevenStructure("Cannot build a structured grid if the chosen element size doesn't fit into the domain!")

    name = 'rect_struct.data'
    if path.endswith('/'):
        filename = path+name
    else:
        filename = path+'/'+name


    elemMap = []
    coordMap = []
    node = 1 #start at node 1, obviously
    for elem in range(1,nelem+1):
        #four nodes in each element for rectangular elements

        elemMap.append([str(elem), str(node), str(node+1), str(node+nx), str(node+nx+1)])
        node +=1
        if node%nx == 0:
            node+=1

    x = xmin
    y = ymin
    for node in range(1,(nx*ny)+1): #unfortunate, but we clobber our previous 'node' variable - don't need it anyway
        coordMap.append([str(node), str(x), str(y)])

        if isEqual(x,xmax, tol=dx/2):
            x = xmin
            y += dy
        else:
            x += dx

        
    with open(filename,'w') as file:
        file.write('#structured rectangles for project 2, PHYS 640\n')
        file.write('#map\n')
        for elem in elemMap:
            line =  ' '.rjust(5).join(elem) + '\n'
            file.write(line)

        file.write('\ncoordinates (x, y)\n')

        for node in coordMap:
            line = ' '.rjust(5).join(node) + '\n'
            file.write(line)

def writeStructTriMap(nx, ny, xmin, xmax, ymin, ymax, dx, dy, path):
    '''Given the number of nodes in the x and y directions, as well as the minimum
and maximum range values in x and y, and the jump steps in x and y, write a structured
triangular map to the given filepath named tri_struct.data'''

    nelem = (nx-1)*(ny-1)*2 #total number of elements
    if (xmax-xmin)%((nx-1)*dx) or (ymax-ymin)%((ny-1)*dy): #This says that if the interval covered by X or Y doesn't fit an even multiple of the number of jumps in that direction, the domain doesn't fit the jumps
        #ny or ny - 1 because we need to consider the nodes at xmin and ymin as well. Effectively nx-1 is number of x elements
        raise UnevenStructure("Cannot build a structured grid if the chosen element size doesn't fit into the domain!")

    name = 'tri_struct.data'
    if path.endswith('/'):
        filename = path+name
    else:
        filename = path+'/'+name


    elemMap = []
    coordMap = []
    node = 1 #start at node 1, obviously
    for elem in range(1,nelem+1):
        #three nodes in each element for triangular elements

        if elem%2: #element number is not cleanly divisible by two, which means a top triangle
            elemMap.append([str(elem), str(node), str(node+nx), str(node+nx+1)])
        else:
            elemMap.append([str(elem), str(node), str(node+1), str(node+nx+1)])
            node +=1
        
        if node%nx == 0:
            node+=1

    x = xmin
    y = ymin
    for node in range(1,(nx*ny)+1): #unfortunate, but we clobber our previous 'node' variable - don't need it anyway
        coordMap.append([str(node), str(x), str(y)])

        if isEqual(x, xmax, tol=dx/2):
            x = xmin
            y += dy
        else:
            x += dx

        
    with open(filename,'w') as file:
        file.write('#structured triangles for project 2, PHYS 640\n')
        file.write('#map\n')
        for elem in elemMap:
            line =  ' '.rjust(5).join(elem) + '\n'
            file.write(line)

        file.write('\ncoordinates (x, y)\n')

        for node in coordMap:
            line = ' '.rjust(5).join(node) + '\n'
            file.write(line)

def getProblem(filename):
    '''Given a filename string, return various problem configuration options from that file in a dict.'''
    with open(filename) as file: #when you use a with statement to open a file, you don't have to close it again, even if the program throws an error
        variables = {}
        for line in file:
            line = line.strip().rstrip() #get rid of hidden whitespace
            if line and not line.startswith('#'):#skip blanks and comments
                varname,value = line.split(':',2)[0:2]
                varname = varname.strip()
                variables[varname] = value
    #clean up a little by replacing references with the referenced value
        for key in variables.keys():
            values = variables[key].split(',')
            for elem in values:
                elem = elem.strip()
                
                if elem in variables.keys():# and elem is not 'T0':
                    newelem = variables[elem]
                    variables[key] = variables[key].replace(elem, newelem)

        for key in variables.keys():
            values = variables[key].split(',')
            i = 0 #index for elements, because I can't figure another way on the fly
            for elem in values:
                elem = elem.strip()
                try:
                    elem = float(elem)
                    
                except ValueError:
                    pass #not able to make it a float, because it's one of the strings that should be there
                values[i] = elem
                i+=1
            variables[key] = values
            

        return variables


##
##
###boundary conditions
##TOP = (MIN_X, Q, 'dirichlet') #takes the form MIN TO START, VALUE OF BOUNDARY, TYPE OF CONDITION; takes multiple definitions in one tuple
##BOTTOM = (MIN_X, 0, 'dirichlet')
##LEFT = (MIN_Y, 0, 'neumann')
##RIGHT = (MIN_Y, 0, 'dirichlet', 1, 0, 'neumann', 2, Q, 'dirichlet')
##
##
###cartesian vectors spanning the problem
##NX = math.ceil((MAX_X-MIN_X)/DX) + 1
##X = [i*DX for i in range(NX+1)]
##
##NY = math.ceil((MAX_Y-MIN_Y)/DY) + 1
##Y = [i*DY for i in range(NY+1)]

def poissonTriElement(elem, elemMap, coords):
    '''Given an element number, a dict which contains the node mapping for that
    element, and a dict which contains the node #'s and their x, y coordinates,
    return the K matrix for that triangular linear element. Used for building K
    for the domain space. Does not assume structure; that is handled by how the
    coords are constructed. Assumes widdershins orientation, starting from the
    bottom left.'''

    #TODO: handle f. Low priority, though, since for our problem f=0 across the whole domain

    localNodeCoords = []
    for node in elemMap[elem]:
        #find the coords of each element
        localNodeCoords.append(coords[node])
        

    ones = np.matrix([[1, 1, 1]])
    rumpA = np.matrix(localNodeCoords).T
    A = np.vstack((ones,rumpA))
    
    
    area = 0.5*abs(np.linalg.det(A)) #area of a triangle with arbitrary vertices
    
    #now find alpha, beta, and gamma - then Ke

    #alpha = [] do I even need alpha? Its submatrix term cancels out.
    beta = []
    gamma = []

    Ke = np.matrix(np.zeros((3,3)))
    
    x = [localNodeCoords[i][0] for i in range(len(localNodeCoords))]
    y = [localNodeCoords[i][1] for i in range(len(localNodeCoords))]

    
    
    for i in range(3):
        j = (i+1)%3 #modulus arithmetic; i = 0 > j = 1; i = 1 > j = 2; i = 2 > j = 0
        k = (j+1)%3 #as above but with j & k. The cheap way to build cyclic variables.
        
        #alpha.append(x[j]*y[k] - x[k]*y[j]) 
        beta.append(y[j] - y[k])
        gamma.append(-(x[j]-x[k]))

    for i in range(3):
        for j in range(3):
            Ke[i,j] = K*((beta[i] * beta[j]) + (gamma[i] * gamma[j]))/(4*area)

        
        fe = (area*F/3)*np.matrix('1;1;1')
        
    return (Ke, fe)
        

def poissonRectElement(elem, elemMap, coords):
    '''Given an element number, a dict which contains the node mapping for that
    element, and a dict which contains the node #'s and their x, y coordinates,
    return the K matrix for that rectangular linear element. Used for building K
    for the domain space. Does not assume structure; that is handled by how the
    coords are constructed. Assumes widdershins orientation, starting from the
    bottom left.'''

    localNodeCoords = []
    for node in elemMap[elem]:
        #find the coords of each element
        localNodeCoords.append(coords[node])

    #this one's much easier than the triangular elements, in a way. We can 'cheat',
    #since rectangles are always right. (I REFUSE TO DO FULL QUADRILATERALS, though I could...)

    x = [localNodeCoords[i][0] for i in range(len(localNodeCoords))]
    y = [localNodeCoords[i][1] for i in range(len(localNodeCoords))]

    a = abs(x[0] - x[1])
    b = abs(y[0] - y[3]) #lengths of the sides. Would be the same if we used the other nodes, since is rectangle\
    #cheaty bit
    s11 = (b/(6*a))*np.matrix('2 -2 -1 1; -2 2 1 -1; -1 1 2 -2; 1 -1 -2 2')
    s22 = (a/(6*b))*np.matrix('2 1 -1 -2; 1 2 -2 -1; -1 -2 2 1; -2 -1 1 2')
    fe = 0.25*F*a*b*np.matrix('1;1;1;1')

    Ke = K*(s11 + s22)

    return (Ke, fe) 

def buildStiffness(elemMap, coords, elemFun): 
    '''Given a map of element nodes and a map of the x,y coordinates of each node,
return a global stiffness matrix for the problem. Boundary conditions are handled
elsewhere; this is just the unmodified matrix equation K*u = Q + f with u unknown.

elemFun is the function used to build local matrices for your desired element type.
Should work regardless of element shape.'''

    #preinitialize, because it makes life easier
    nodeQuantity = len(coords) #how many nodes there are
    stiff = np.matrix(np.zeros((nodeQuantity,nodeQuantity))) #K matrix is (nodexnode) rows and columns - square
    f = np.matrix(np.zeros((nodeQuantity,1))) #f vector
    
    for elem in elemMap:
        Ke, fe = elemFun(elem, elemMap, coords)
        #print(elem, Ke)
        m = 0
        n = 0
        for row in elemMap[elem]: #one of two primary debug areas; the other is the BC

            for column in elemMap[elem]:
                stiff[row-1,column-1] += Ke[m,n] #here we're being punished for changing our indexing
                n +=1

            f[row-1] += fe[m]
            n = 0
            m+=1
        
    
    return stiff, f #stiff

def boundFun(x, axis='x'):
    '''Given an x coordinate, return the value of the boundary function as
defined in the project.'''

    #second argument is for generality, but not mentioned in the docstring because
    #we will never use it

    if axis == 'x':
        a = AX
    else:
        a = AY
    T = T0*(math.cos((math.pi*x)/(8*a)))

    return T

def applyBounds(stiff, f, coords):
    '''Apply the boundary values for the problem, defined in the problem.cfg file.
Note that this relies on module-level variables defined at the bottom of this library.'''
    #first, check the boundary conditions, so dirichlet gets priority
    q = np.matrix(np.zeros(len(stiff))).T #a list which will hold our known q values, plus their indices. A list of tuples.
    topNodes = []
    bottomNodes = []
    leftNodes = []
    rightNodes = []
    
    for node in coords: #this tree is VERY sensitive. It works for the given problem config, but not necessarily any others. Fix to work properly if I get time.

        #a fix would involve checking NOW to see which are dirichlet and which are neumann conditions, and transforming the bounds lists to privilege dirichlet
        #basically the problem I'm avoiding here is applying the neumann bcs and getting corners that are way off
        if isEqual(coords[node][1],MIN_Y):
            bottomNodes.append(node)
        elif isEqual(coords[node][0],MAX_X):
            rightNodes.append(node)
        elif isEqual(coords[node][1],MAX_Y):
            topNodes.append(node)
        elif isEqual(coords[node][0],MIN_X):
            leftNodes.append(node)

    removeNode = [] #an index of each node number to remove (i.e., column in stiffness matrix). A tuple; the first value is the node number, and the second is the value of that node number.
    knownQ = [] #an index of each resultant vector row to remove
    #now we've found all the nodes on the boundaries, so we can transform our stiffness matrix

    #first, figure out which boundary conditions should apply! and apply them to q

    for node in topNodes:
        pos = coords[node][0] #crawl along x
        value, kind = findBound(pos, TOP)
        if value == 'boundfun':
            value = boundFun(pos)

        if kind == 'neumann': #yay, it's easy! though for this project it's actually synonymous with a no-op        
            knownQ.append((node, value)) #say we know what Qi is, where i is the node number, and say what that Qi equals
            #then leave our prospective U's alone (i.e., don't reshape the stiffness matrix)

        elif kind == 'dirichlet': #we know what our node value is for this node
            removeNode.append((node, value))

        else:
            raise ValueError('Boundaries improperly specified. Please check the problem.cfg file.')
            
    for node in bottomNodes:
        pos = coords[node][0] #crawl along x
        value, kind = findBound(pos, BOTTOM)
        if value == 'boundfun':
            value = boundFun(pos)

        if kind == 'neumann': #yay, it's easy! though for this project it's actually synonymous with a no-op        
            knownQ.append((node, value)) #say we know what Qi is, where i is the node number, and say what that Qi equals
            #then leave our prospective U's alone (i.e., don't reshape the stiffness matrix)

        elif kind == 'dirichlet': #we know what our node value is for this node
            removeNode.append((node, value))

        else:
            raise ValueError('Boundaries improperly specified. Please check the problem.cfg file.')

    for node in leftNodes:
        pos = coords[node][1] #crawl along y
        value, kind = findBound(pos, LEFT)
        if value == 'boundfun':
            value = boundFun(pos)

        if kind == 'neumann': #yay, it's easy! though for this project it's actually synonymous with a no-op        
            knownQ.append((node, value)) #say we know what Qi is, where i is the node number, and say what that Qi equals
            #then leave our prospective U's alone (i.e., don't reshape the stiffness matrix)

        elif kind == 'dirichlet': #we know what our node value is for this node
            removeNode.append((node, value))

        else:
            raise ValueError('Boundaries improperly specified. Please check the problem.cfg file.')

    for node in rightNodes:
        pos = coords[node][1] #crawl along y
        value, kind = findBound(pos, RIGHT)
        if value == 'boundfun':
            value = boundFun(pos)

        if kind == 'neumann': #yay, it's easy! though for this project it's actually synonymous with a no-op        
            knownQ.append((node, value)) #say we know what Qi is, where i is the node number, and say what that Qi equals
            #then leave our prospective U's alone (i.e., don't reshape the stiffness matrix)

        elif kind == 'dirichlet': #we know what our node value is for this node
            removeNode.append((node, value))

        else:
            raise ValueError('Boundaries improperly specified. Please check the problem.cfg file.')


    #now that we've mapped out the boundary conditions, we can FINALLY apply them

    for pair in knownQ: #assign the known values of q to the appropriate places
        qi, value = pair
        q[qi-1] = value

    #now combine q and f, since we don't actually care about q values

    q += f

    #finally, trawl through the proper rows and columns of the stiffness matrix to remove known node values

    #we want to go through our matrix in reverse, so we aren't clobbering our own indices
    removeNode.sort(reverse=True) #take that, Matlab



    #Seems to maybe be a problem with BCs? I get very bad value for node 1 for instance
    
    for pair in removeNode:
        node, value = pair

        addToQ = stiff[:,node-1]*value
        
        q-= addToQ#stiff[:,node-1]*value #for all q, add node*(coeff in stiffness matrix) for known node for all coupled equations.
                
        stiff = np.delete(np.delete(stiff, node-1, 1), node-1, 0) #delete row for known node and column for known node - 'cause we know it

        q = np.delete(q, node-1, axis=0) #delete the row in q corresponding to the known node value
        
    
    #from this function, we want the reduced stiffness matrix and q ( = q + f),
    #AND the list of nodes we deleted and their values, so we can wrap them back in later

    return removeNode, stiff, q

def findBound(pos, boundary):
    '''Given a boundary tuple and an x or y position, return which condition actually applies
    pos is a position in x or y
    boundary is a tuple in the form (MIN, BOUNDARY_VALUE, TYPE), that can hold any number of boundary conditions
        MIN is the minimum pos where the condition applies
        BOUNDARY_VALUE is the actual value of the boundary condition
        TYPE is the type of boundary condition, either 'dirichlet' or 'neumann'
    returns a tuple in the form of (VAL, TYPE) where VAL is the value of the boundary condition and TYPE is the type, either neumann or dirichlet'''

    for i in range(len(boundary)-3,-1,-3): #go through the boundary tuple backward
        if pos < boundary[0]:
            pos = boundary[0] #catch rounding errors
        if pos >= boundary[i]:
            return(boundary[i+1], boundary[i+2])

def solveU(knownNodes, reducedStiff, q):
    '''Given a list of tuples in removeNode that includes known nodes and their
values, the reduced stiffness matrix, and the corresponding RHS vector, solve
for the vector of nodes, then re-introduce the known values in the appropriate places.'''

    knownNodes.sort()
    u = np.linalg.solve(reducedStiff, q) #u is our solution vector

    
    for pair in knownNodes:
        node, val = pair
        u = np.insert(u, node-1, val, axis=0)

    return(u)
        

def analyticTemp(x, y):
    '''Given a point specified by (x, y), return the value of T at that point,
assuming the function (and therefore the bounds) specified in PHYS 640, Project 2.'''

    pi = math.pi

    sinhtop = math.sinh(pi*y/8*AY)
    costop = math.cos(pi*x/8*AX)

    top = T0*sinhtop*costop

    bottom = math.sinh(pi/2)

    T = top/bottom

    return T
    
##def yComp(psi, i, j, dx):
##    '''Given a matrix psi that represents the point solutions to some function psi(x,y),
##    and both a row and column index, return the point solution
##    V = -dpsi/dx, the y-component of the vector field at that point
##    dx:    the step size in x'''
##
##    V = -(psi[i][j+1] - psi[i][j-1])/(2*dx)
##    return V
##
##def xComp(psi, i, j, dy):
##    '''Given a matrix psi that represents the point solutions to some function psi(x,y),
##    and both a row and column index, return the point solution
##    U = dpsi/dy, the x-component of the vector field at that point
##    dy:    the step size in y'''
##    U = (psi[i+1][j] - psi[i-1][j])/(2*dy)
##    return U
##
##def vectorField(psi,dx,dy):
##    '''Given a matrix psi that represents the point solutions to some function psi(x,y), return
##a matrix that represents the vector field for that function.
##        Note that the returned matrix will be 1 smaller in both dimensions than psi
##        dx: the step size in x of the psi matrix
##        dy; the step size in y of the psi matrix'''
##    m = len(psi) - 2
##    n = len(psi[0]) - 2
##    U = [[0 for i in range(n)] for j in range(m)]
##    V = [[0 for i in range(n)] for j in range(m)]
##    
##    for i in range(1,m):
##        for j in range(1,n):
##            U[i][j] = xComp(psi, i, j, dy)
##            V[i][j] = yComp(psi, i, j, dx)
##            
##    
##    return (U,V)

def isEqual(i,j,tol=1e-15):
    '''Given two numbers i & j, return True if they are within tol of each other, otherwise return False.'''
    i,j = abs(i),abs(j)
    diff = abs(i-j)
    return (diff<=tol)


#down here at the bottom we're going to set up the module-level variables

VARIABLES = getProblem('problem.cfg')

#x and y config
MIN_X = VARIABLES['min_x'][0]
MIN_Y = VARIABLES['min_y'][0]
MAX_X = VARIABLES['max_x'][0]
MAX_Y = VARIABLES['max_y'][0]
DX = VARIABLES['dx'][0]
DY = VARIABLES['dy'][0]
NX = math.ceil((MAX_X-MIN_X)/DX) +1
X = [i*DX for i in range(NX+1)]
NY = math.ceil((MAX_Y-MIN_Y)/DY) +1
Y = [i*DY for i in range(NY+1)]

AX = (MAX_X-MIN_X)/4 #domain goes from (0*a + x0) to (4*a +x0)
AY = (MAX_Y-MIN_Y)/4


#boundaries
TOP = VARIABLES['TOP']
BOTTOM = VARIABLES['BOTTOM']
RIGHT = VARIABLES['RIGHT']
LEFT = VARIABLES['LEFT']
T0 = VARIABLES['T0'][0]

#coefficients
K = -VARIABLES['k'][0]
F = VARIABLES['f'][0]


#how will we solve?
#SOLVER = VARIABLES['solver'][0]
#TOL = VARIABLES['tol'][0]
#MAXITERS = VARIABLES['maxiters'][0]

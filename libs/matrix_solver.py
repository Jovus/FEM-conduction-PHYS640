#TODO: Break this out into its own library, then
#put the import statements in the runfile. But this works for testing
import os,sys,numpy
import scipy.io as sio
##path = os.getcwd() + '/libs' 
##sys.path.append(path) #add our custom libs directory to the $PYTHONPATH
try:
    import matrix_manipulation as mman #add custom modules to this line
except ModuleNotFoundError:
    print('Please only run this from the matrix_solver directory.')
    raise ModuleNotFoundError
    sys.exit()


def readMatrix(source=''):
    '''Given a path to a Matrix Market file in string format, reads it in as a
matrix. Using this function instead of just scipy.io.mmread also strips the matrix
down into a nested list.'''
    validpath=False
    while not validpath:
        try:
            mat = sio.mmread(source)
            validpath=True
        except FileNotFoundError:
            source=input('Please enter a valid filepath to a matrix market file: ').strip("'")
            validpath=False
    mat = mat.tolist() #we don't want the full matrix object functionality
    return mat

def writeMatrix(mat,file):
    '''Given a matrix and a filename, writes to a Matrix Market file at file.
Using this function instead of just scipy.io.mmwrite because we aren't using
matrix objects.'''
#TODO: Define default path
    mat = numpy.matrix(mat)
    sio.mmwrite(file,mat)
    return None

def lowerSolve(A,B):
    '''Given a lower triangular mxn matrix A and an mx1 vector B, returns x
according to Ax = B. Doesn't check lengths.'''
    m,n = len(A),len(A[0]) #rows, columns
    x = [[1] for i in A] 
    for i in range(m):
        diff = B[i][0] #we will subtract from this

        #subtract what we know already
        for j in range(i): #rows
            #if j >= 0:
            diff -= A[i][j]*x[j][0]

        #now x = B/A
        x[i][0] = diff/A[i][i]

    return x

def swapRows(A,B,row1, row2):
    '''Given A and B in equation Ax = B and two 0-indexed row numbers, swaps their locations in the matrix.'''
    A[row1],A[row2] = A[row2],A[row1]
    B[row1],B[row2] = B[row2],B[row1]
    return None

def multRows(A, B, row, mult):
    '''Given A and B in equation Ax = B, multiply row (specified by zero-indexed rownumber) by scalar mult.'''
    A[row]= [mult*i for i in A[row]]
    B[row] = [mult*i for i in B[row]]
    return None

def addRows(A,B,row1,row2):
    '''Given A and B in equation Ax = B, add row1 to row2 and store in row1. row1 and row2 arguments are indices, starting with 0.
Changes row 1'''
    A[row1] = [i + j for i,j in zip(A[row1],A[row2])]
    B[row1] = [B[row1][0] + B[row2][0]] #don't need the extra overhead of zip()
    return None

def isEqual(i,j,tol=1e-15):
    '''Given two numbers i & j, return True if they are within tol of each other, otherwise return False.'''
    i,j = abs(i),abs(j)
    diff = abs(i-j)
    return (diff<=tol)

def isLower(A, tol=1e-15):
    '''Given a matrix A, check to see if it is lower triangular'''
    m, n = len(A),len(A[0]) #number of rows and columns
    for i in range(m):
        for j in range(i+1,n):
            lower = isEqual(A[i][j],0,tol)
            if not lower:
                return lower

    return lower

def decompMatrix(A):
    '''Given a square matrix A, return its Diagonal, Upper, and Lower component matrices'''
    n = len(A)
    
    D = []
    L = []
    U = []
    for i in range(n): #because if you try any of the slick ways, you end up with nested list references, which is a really huge pain
        D.append([0]*n)
        U.append([0]*n)
        L.append([0]*n)
    for i in range(n):
        D[i][i] = A[i][i]

        for j in range(n-1,i-1,-1):
            U[i][j] = A[i][j]

    for i in range(n-1,-1,-1):
        for j in range(i+1):
            L[i][j] = A[i][j]

    return D,U,L

def infNorm(A):
    '''Given a matrix (or vector) A, return the infinity-norm.'''
    inf = 0
    for row in A:
        for elem in row:
            elem = abs(elem)
            if elem > inf:
                inf = elem

    return inf

def gaussSolve(A,B, tol=1e-15): 
    '''Given a matrix A and a vector B for equation Ax = B, find x via naive
Gaussian elimination'''
    lower = isLower(A,tol) #first, check to see if we need to do anything
    #if we do, drop into the main solver loops after setting up the indices
    m,n = len(A),len(A[0])
    
    #main solver loops
    for j in range(n-1,-1,-1):  #go backward through the columns
        for i in range(m-1): #ignore the bottom row
            if isEqual(A[i+1][j],0,tol) or isEqual(A[i][j],0,tol):
                continue
            mult = -(A[i+1][j]/A[i][j])
            multRows(A,B,i,mult)
            addRows(A,B,i,i+1)
            if isEqual(A[i][j],0,tol):
                A[i][j] = 0 #get rid of (what we hope is just) rounding errors
            #print('Row {0} = {1}\n'.format(i,A[i]))
        m -=1
    #now actually solve the matrix
    x = lowerSolve(A,B)
    return x

def pivotSolve(A,B,tol=1e-15):
    '''Given a matrix A and a vector B for equation Ax = B, find x via partial
pivot elimination'''
    lower = isLower(A,tol) #first, check to see if we need to do anything
    m,n = len(A),len(A[0])

    for j in range(n-1,-1,-1): #loop over the columns backward to find a pivot
        p = j #first, we assume that the pivot is in the last row

        for i in range(j):
            if abs(A[i][j]) < abs(A[p][j]):
                p = i #find the biggest element in the chosen column

        if p is not j:
            swapRows(A,B,p,j) #put the pivot at the bottom

        #now do gaussian elimination
        for i in range(j):
            mult = -(A[i+1][j]/A[i][j])
            if isEqual(mult,0,tol):
              continue #don't multiply by zero
            multRows(A,B,i,mult)
            addRows(A,B,i,i+1)
            if isEqual(A[i][j],0,tol):
                A[i][j] = 0

    x = lowerSolve(A,B)
    return x

def jacobiSolve(A,B,tol=1e-15):
    '''Given a square matrix A and a vector B for equation Ax=B, applies the Jacobi
iterative method to solve for x'''
    n = len(A) #matrix dimension (square)
    err = 1
    iterations = 0
    infB = infNorm(B)
    x0 = [[1] for i in range(n)] #initial guess for x
    xk = [[0] for i in range(n)] #preallocate 'next guess' vector

    while err > tol:
        #iterations += 1
        if iterations > 500:
            break
        for i in range(n):
            s = 0
            for j in range (n):
                if j is not i:
                    s += A[i][j]*x0[j][0]
            xk[i][0] = (B[i][0]-s)/A[i][i]

        err = infNorm(mman.matrixDiff(xk,x0))
        x0 = [[xk[i][0]] for i in range(len(xk))]

    return x0

def conjGradSolve(A,B,tol=1e-8,maxiters=1e6):
    '''Given a square matrix A and a vector B for equation Ax=B, use the conjugate
gradient method to solve for x.'''
    n = len(A)

    x0 = [[1] for i in range(n)] #initial guess for x
    xk = [[0] for i in range(n)] #preallocate 'next guess' vector
    aug = mman.vectMatMult(A,x0)
    res = mman.matrixDiff(B,aug) #calculate the initial residue
    iters = 0
    if iters == 0 and abs(infNorm(res)) <=tol: #we're done; lucky guess
        print('We got a lucky first guess and need do no iterations.')
        return x0
    d = [[elem for elem in row] for row in res]
    deltak = mman.dotProduct(res,res) #delta k is a number
    delta0 = deltak
    err = 1
    while iters<maxiters and err > tol:
        
        q = mman.vectMatMult(A,d) #q is a vector
        alpha = deltak/mman.dotProduct(d,q) #alpha is the line search scalar
        xk = mman.matrixAdd(x0,mman.scaleMat(d,alpha)) #new x = old x + alphad
        if iters%5==0:
            #reset the remainder after ten iterations to avoid picking up too much error
            aug = mman.vectMatMult(A,xk)
            res = mman.matrixDiff(B,aug)
        else:
            res = mman.matrixDiff(res, mman.scaleMat(q,alpha))

        #now set things up for the next iteration
        delta0 = deltak
        deltak = mman.dotProduct(res,res)
        beta = deltak/delta0 #beta is a number
        d = mman.matrixAdd(res, mman.scaleMat(d,beta))
        err = infNorm(res)
        x0 = [[elem for elem in row] for row in xk]
        iters +=1
        if iters%1000 == 0:
            print('Number of iterations = {}'.format(iters))
        
    print(iters)
    return x0

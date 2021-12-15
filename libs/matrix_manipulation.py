#Matrix manipulations using tuples. Scipy and numpy have a native
#matrix class for use with their various linear algebraic solvers,
#but that's way too heavy duty for our work here

#Includes functions for a vector sum, dot product, 3d cross product,
#magnitude of a vector, normalizing a vector, multiplying vectors,
#multiplying a vector by a matrix, and matrix multiplication

import math

def getMatrix():
    '''Get a matrix from the user in (1,2,3;4,5,6) format. Notice this will only
work for m x n matrices. Example given is a 2 x 3 matrix.'''
    M = input('''Please input a matrix in the form (a, b, c; d e f) [example is a 2x3 matrix].
If you want to input a vector, just use a matrix with only one column [e.g. (1;2;3).\n''')

    M = M.strip('([])').split(';')
    Mp = [] #M prime, a temporary holding list
    for i in M:
        i = i.split(',')
        i = [int(x) for x in i]
        Mp.append(i)

    M = Mp 
    #TODO: Add some error checking so that we know we're getting a well-formed matrix
    #or vector, and the user hasn't fat-fingered the keys. Not important for right now, though
    printMatrix(M)
    return(M)

def vectorLength(A):
    """Given a vector A, return its length, as a number."""
    length = 0
    
    for i in range(len(A)):
        length += math.pow(A[i][0],2) #add the squares of the elements together

    length = math.sqrt(length) #and then take the square root.
    return (length)

def vectorNorm(A):
    """Given a vector A, return the normal vector that corresponds to it."""
    divisor = vectorLength(A)
    C = [[i[0]/divisor] for i in A]
    return (C)

def vectorSum(A,B):
    """Given two vectors A and B, return the sum."""

    A,B = agreeVectors(A,B) #make the lengths equal by appending zeros to the shorter
    C = [[A[i][0] + B[i][0]] for i in range(len(A))]

    return (C)


def agreeVectors(A,B):
    """Given two vectors A and B, append 0's to the shorter one so their length is the same.
Returns both vectors as a tuple, so proper assignment looks like X,Y = agreeVectors(A,B)."""
    
    la = len(A)
    lb = len(B)
    i = la - lb #negative if B is longer

    if i > 0: #A is longer
        for j in range(i):
            B.append([0])
    elif i < 0: #B is longer
        for j in range(-i):
            A.append([0])
    else:
        pass #equal length

    return(A,B)        
    
def dotProduct(A,B):
    '''A and B are vectors. Returns the dot product A.B'''
    A,B = agreeVectors(A,B)
    length = len(A)
    C = 0
    for i in range(length):
        C += A[i][0] * B[i][0]
    return (C)

def crossProduct(A,B):
    '''A and B are three-dimensional vectors. Returns the cross product of AxB'''

    if len(A) != 3 or len(B) != 3:
        print("One of the supplied vectors is not three-dimensional. I can't do anything here.\n")
        return (None)

    #might do this more easily by just calling a determinant function, but this works for now
    
    I = A[1][0]*B[2][0] - A[2][0]*B[1][0] #first term of the cross product
    J = A[2][0]*B[0][0] - A[0][0]*B[2][0] #second
    K = A[0][0]*B[1][0] - A[1][0]*B[0][0] #third
    
    C = ([I],[J],[K])
    return (C)
    
def vectorProduct(A,B):
    """Given two vectors of equal length A and B, return their matrix product."""
    #fair bit of rewrite code in here from matrixMult. Could probably clean that up
    #with a call to matrixMult (or a staging function)

    #preallocate the return matrix
    
    m = range(len(A)) #treat A like column vector
    n = range(len(B)) #treat B like row vector
    C = [[0 for i in m] for j in n]

    for i in m:
        for j in n:
            C[i][j] = A[i][0]*B[j][0]

    return(C)

def multCheck (A,B):
    """Given two matrices A and B, check to see if they can be multiplied."""
    if len(A) != len(B[0]):
        return (False)
    else:
        return (True)
    
def vectMatMult(A,B):
    """Given a vector B and a matrix A, return their product AB. Returns type None
if multiplication is undefined."""

    canMult = multCheck(B,A)
    if not canMult:
        print('This vector cannot be multiplied by this matrix.\n')
        return (None)

    #C = [[A[i][0]*B[i][j] for j in range(len(B[0]))] for i in range(len(A))] list comprehensions are great. Reading comprehension is greater.
    C = [[0] for elem in B]
    for i in range(len(B)):
        s = 0
        for j in range(len(A)):
            s+= B[i][0] * A[i][j]

        C[i][0] = s
    return C

def matrixMult(A,B):
    """Given matrices A and B, returns their product AB. Returns type None if matrices cannot be multiplied.
    Do not use for vector direct product; call vectorProduct instead."""
    
    canMult = multCheck(A,B)
    if not canMult:
        print('Matrices cannot be multiplied. Dimensions do not match.\n')
        return (None)

    #preallocate the return matrix
    
    m = range(len(A)) #rows in A
    n = range(len(B[0])) #columns in B
    C = [[0 for i in m] for j in n]
    
    for i in range(len(A)): #rows of C
        for j in range(len(B[0])): #columns of C
            for k in range(len(A[0])): #decoupled 
                try:
                    C[i][j] += A[i][k]*B[k][j]
                except IndexError: #catch non-square matrices. Could probably make some complicated construct to save cycles, but that might actually be more expensive than just iterating through the loops dumbly.
                    C[i][j] += 0
                
    return(C)

def matrixAdd(A,B):
    '''Given matrices A and B, returns their sum A+B. Returns type none if matrix dimensions do not match.'''
    m,n = len(A),len(A[0])
    o,p = len(B),len(B[0])
    if m is not o or n is not p:
        return None
    
    s = [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return s

def matrixDiff(A,B):
    '''Given matrices A and B, returns their difference A-B. Returns type none if matrix dimensions do not match.'''
    m,n = len(A),len(A[0])
    o,p = len(B),len(B[0])
    if m != o or n != p:
        print('Matrix dimensions do not match')
        return None
    
    diff = [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return diff

def transformMat(A):
    '''Given a matrix A, return its transform.'''
    m,n = len(A),len(A[0])
    T = [[0 for j in range(m)] for i in range(n)]
    for i in range(m):
        for j in range(n):
            T[j][i] = A[i][j]

    return T

def scaleMat(A,s):
    '''Given a matrix A and a scalar s, multiply A by s.'''
    T = [[elem*s for elem in row] for row in A]
    return T

def goAgain():
    again = input("Would you like to perform another operation? (y or n)\n")
    return (again.lower().startswith('y'))

def printMatrix(A):
    """Handed a matrix, will print it to the screen in a hopefully not terribly ugly manner."""

    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in A]))


##doOps = True
##
##print("Please note matrices and vectors MUST be well-formed. If you try to get tricky, the program will fail.\n")
##
##while (doOps):
##    op = input("""What would you like to do? You can find the [l]ength of a vector, [n]ormalize a vector, [s]um two vectors, take their [d]ot product, find their [c]ross product (for three-dimensional vectors only), find their vector [p]roduct, multiply a [v]ector by a matrix, or [m]ultiply two matrices:\n""")
##    op = op.lower()[0]
##
##    #main decision loop to choose betweeen ops
##
##    #first cases: only need one input
##    if op in 'l':
##        A = getMatrix()
##        length = vectorLength(A)
##        print('Length of that vector is {0:.5f}\n'.format(length))
##
##    elif op in 'n':
##        A = getMatrix()
##        vector = vectorNorm(A)
##        print('Norm of the matrix is:\n')
##        printMatrix(vector)
##
##    #now we need two inputs
##    elif op in 's':
##        A = getMatrix()
##        B = getMatrix()
##        vector = vectorSum(A,B)
##        print('The sum of the vectors is:\n')
##        printMatrix(vector)
##
##    elif op in 'd':
##        A = getMatrix()
##        B = getMatrix()
##        dot = dotProduct(A,B)
##        print('The dot product of the vectors is: {}'.format(dot))
##
##    elif op in 'c':
##        A = getMatrix()
##        B = getMatrix()
##        cross = crossProduct(A,B)
##        print('The cross product of the vectors is:\n')
##        printMatrix(cross)
##
##    elif op in 'p':
##        A = getMatrix()
##        B = getMatrix()
##        product = vectorProduct(A,B)
##        print('The product of the vectors is:\n')
##        printMatrix(product)
##
##    elif op in 'v':
##        A = getMatrix()
##        B = getMatrix()
##        mult = vectMatMult(A,B)
##        print('The product of the vector and the matrix is:\n')
##        printMatrix(mult)
##
##    elif op in 'm':
##        A = getMatrix()
##        B = getMatrix()
##        mult = matrixMult(A,B)
##        print('The product of the matrices is:\n')
##        printMatrix(mult)
##
##    elif op in 'aq':
##        break
##    else:
##        print("I didn't understand your input. Please try again, or type [a] or [q] to quit.\n")
##        continue
##
##
##    doOps = goAgain()
##
##    

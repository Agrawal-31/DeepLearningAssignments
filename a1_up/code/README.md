## Matrix class usage examples

```cpp
// Creates a matrix with 2 rows and 3 columns
// All elements are stored as floating point numbers (currently long double)
Matrix a(2, 3);

// Access and modify individual members using [][]
a[0][0] = 1.2;

// Set all elements to 0
a = {};

// Set all elements to 12
a = {12}

// Assign all elements in one statement
a = {8.9, 2, 3,
       4, 5, 6};


// Initialization with matrices
Matrix b = a, c;

// Assignment of one matrix to another 
c = a;

// Addition, subtraction, multiplication and transpose
// transpose returns a tranposed matrix, doesn't modify the matrix on which
// it is called
c = (a + b - c) * c.transpose();

// Matrices can be printed
cout << c;

// Multiplication and division with scalars works too
c = ((2 * a) + (a * 2)) / 4;

// This returns a new matrix containing the inverse. Doesn't modify the
// existing matrix itself
cout << a.inverse();

// Calculate determinant
cout << a.determinant();

// initialise Matrix with all elements 1
a.initialize();

// find mean of a matrix
b = a.mean();

// For row or column matrices, the norm (square root of sum of squares of 
// elements) can be computed
Matrix d(1, 3);
d = {1, 2, 3};
cout << d.norm();
```
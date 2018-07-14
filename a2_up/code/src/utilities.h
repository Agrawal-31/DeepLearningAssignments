#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <bits/stdc++.h>

using namespace std;

typedef double ld;
typedef long long ll;

class Matrix {
public:
  // Default constructor
  Matrix();

  // Constructs a matrix with dimensions ('nrows', 'ncols')
  Matrix(ll nrows, ll ncols);

  // Copy constructor
  Matrix(const Matrix& m);

  // Allows access to individual elements
  vector<ld>& operator[](ll row);

  // Assigns the matrix from another
  Matrix& operator=(const Matrix& m);

  // Allows assignments of the form m = {1, 2, 3, 4}
  Matrix& operator=(const initializer_list<ld>& l);

  // Addition assignment with a matrix 
  Matrix& operator+=(const Matrix& m);

  // Subtraction assignment with a matrix 
  Matrix& operator-=(const Matrix& m);
  
  // Multiplication assignment with a matrix 
  Matrix& operator*=(const Matrix& m);
  
  // Multiplication assignment with a number 
  Matrix& operator*=(ld a);
  
  // Division assignment with a number
  Matrix& operator/=(ld a);

  // Compute norm of the row or column matrix
  ld norm();

  // Returns transpose of the matrix
  Matrix transpose();


  // Returns a matrix containing column means
  Matrix mean();
  
  // Sets all elements to 1
  Matrix initializer();

  // Returns the determinant of the matrix
  ld determinant();

  // Returns the inverse of the matrix
  Matrix inverse();

  Matrix apply(ld (*func)(ld), bool in_place = false);
  
  // Append a row before row with index 'before' and set all its elements
  // to 'fill_val'
  void addRow(ll before, ll fill_val);

  // Returns the number of rows
  ll getNumRows() const;

  // Returns the number of columns
  ll getNumCols() const;

  friend Matrix hadamard_prod(const Matrix& m1, const Matrix& m2);
  
  // Returns a matrix representing the sum of two matrices
  friend Matrix operator+(const Matrix& m1, const Matrix& m2);

  // Returns a matrix representing the difference of two matrices
  friend Matrix operator-(const Matrix& m1, const Matrix& m2);

  // Returns a matrix representing the product of two matrices
  friend Matrix operator*(const Matrix& m1, const Matrix& m2);

  // Applicable when 'm' is (1, 1) matrix. Returns the sum of the number
  // and the matrix
  friend ld operator+(const Matrix& m, ld num);

  // Applicable when 'm' is (1, 1) matrix. Returns the sum of the number
  // and the matrix
  friend ld operator+(ld num, const Matrix& m);
  
  // Applicable when 'm' is (1, 1) matrix. Returns the difference of the number
  // and the matrix
  friend ld operator-(const Matrix& m, ld num);
  
  // Applicable when 'm' is (1, 1) matrix. Returns the difference of the number
  // and the matrix
  friend ld operator-(ld num, const Matrix& m);

  // Applicable when 'm' is (1, 1) matrix. Returns the sum of the number
  // and the matrix
  friend ld& operator+=(ld& num, const Matrix& m);
  
  // Applicable when 'm' is (1, 1) matrix. Returns the difference of the
  // number and the matrix
  friend ld& operator-=(ld& num, const Matrix& m);
  
  // Applicable when 'm' is (1, 1) matrix. Returns the product of the
  // number and the matrix
  friend ld& operator*=(ld& num, const Matrix& m);
  
  // Applicable when 'm' is (1, 1) matrix. Returns the division of the
  // number and the matrix
  friend ld& operator/=(ld& num, const Matrix& m);
  

  // Returns a matrix which is the scalar multiplication of 'm' with 'num'
  friend Matrix operator*(const Matrix& m, ld num);

  // Returns a matrix which is the scalar multiplication of 'm' with 'num'
  friend Matrix operator*(ld num, const Matrix& m);
  
  // Returns a matrix which is the division of 'm' with 'num'
  friend Matrix operator/(const Matrix& m, ld num);

  // Used to print the matrix
  friend ostream& operator<<(ostream &out, const Matrix& m);

private:
  // Computes the elements of the submatrix defined by 'row_mask' and 
  // 'col_mask'
  ld determinant(vector<bool>& row_mask, vector<bool>& col_mask);
  
  // Multiplies each element of the matrix with 'num'
  void element_wise_multiply(ld num);

  // Divides each element of the matrix with 'num'
  void element_wise_divide(ld num);
  
  // Stores the elements of the matrix
  vector<vector<ld>> data;
  ll nrows;
  ll ncols;
};

class DataSet {
public:
  // 'file' is the path to the file from where the dataset
  // needs to be fetched
  DataSet(const string& file);
  
  // Prints the dataset
  void print();

  // Stores the instances of dataset in a column matrix
  vector<Matrix> instances;

  // Stores the target class corresponding to each instance
  vector<short> target_class;
};

class Stats {
public:
  // 'tp', 'tn', 'fp', 'fn' represents the number of true positives,
  // true negatives, false positives, false negatives respectively 
  Stats(ll tp, ll tn, ll fp, ll fn);

  // Prints the statistics
  friend ostream& operator<<(ostream& out, const Stats& s);
private:
  ll tp, tn, fp, fn;
  ld precision, recall, accuracy, f_measure;
};

#endif
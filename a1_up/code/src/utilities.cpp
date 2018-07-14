#include <bits/stdc++.h>
#include "utilities.h"

using namespace std;

typedef long double ld;
typedef long long ll;

// Default constructor
Matrix::Matrix() {
  data.resize(1);
  data[0].resize(1);
  nrows = 1; ncols = 1;
}

// Constructs a matrix with dimensions ('nrows', 'ncols')
Matrix::Matrix(ll nrows, ll ncols) {
  if (nrows == 0 || ncols == 0) {
    cerr << "Trying to create matrix of dimension (" << nrows << ", " << ncols;
    cerr << "): Neither dimension is allowed be of size 0\n";
    exit(0);
  }
  data.resize(nrows);
  for (ll i = 0; i < nrows; i++) {
    data[i].resize(ncols);
  }
  this->nrows = nrows;
  this->ncols = ncols;
}

// Copy constructor
Matrix::Matrix(const Matrix& m) {
  data = m.data;
  nrows = m.nrows;
  ncols = m.ncols;
}

// Allows access to individual elements
vector<ld>& Matrix::operator[](ll row) {
  return data[row];
}

// Assigns the matrix from another
Matrix& Matrix::operator=(const Matrix& m) {
  data = m.data;
  nrows = m.nrows;
  ncols = m.ncols;
  return *this;
}

// Allows assignments of the form m = {1, 2, 3, 4}
Matrix& Matrix::operator=(const initializer_list<ld>& l) {
  if (l.size() == 0) {
    for (ll i = 0; i < nrows; i++) {
      for (ll j = 0; j < ncols; j++) {
        data[i][j] = 0;
      }
    }
  } else if (l.size() == 1) {
    for (ll i = 0; i < nrows; i++) {
      for (ll j = 0; j < ncols; j++) {
        data[i][j] = *l.begin();
      }
    }
  } else if (nrows * ncols != static_cast<ll>(l.size())) {
    cerr << "Dimension mismatch: " << nrows * ncols << " != " << l.size() << "\n";
    exit(0);
  } else {
    initializer_list<ld>::iterator it = l.begin();
    for (ll i = 0; i < nrows; i++) {
      for (ll j = 0; j < ncols; j++) {
        data[i][j] = *(it++);
      }
    }
  }
  return *this;
}

// Addition assignment with a matrix 
Matrix& Matrix::operator+=(const Matrix& m) {
  *this = *this + m;
  return *this;
}

// Subtraction assignment with a matrix 
Matrix& Matrix::operator-=(const Matrix& m) {
  *this = *this - m;
  return *this;
}

// Multiplication assignment with a matrix 
Matrix& Matrix::operator*=(const Matrix& m) {
  *this = *this * m;
  return *this;
}

// Multiplication assignment with a number 
Matrix& Matrix::operator*=(ld a) {
  *this = *this * a;
  return *this;
}

// Division assignment with a number
Matrix& Matrix::operator/=(ld a) {
  *this = *this / a;
  return *this;
}

// Compute norm of the row or column matrix
ld Matrix::norm() {
  if (nrows > 1 && ncols > 1) {
    cerr << "Dimension error (" << nrows << ", " << ncols << "): ";
    cerr << "norm() can be computed only on a row or column matrix\n";
    exit(0);
  }
  ld ans = 0;
  for (ll i = 0; i < nrows; i++) {
    for (ll j = 0; j < ncols; j++) {
      ans += data[i][j] * data[i][j];
    }
  }
  return sqrt(ans);
}
  
// Returns transpose of the matrix
Matrix Matrix::transpose() {
  Matrix ans(ncols, nrows);
    for (ll i = 0; i < ans.nrows; i++) {
      for (ll j = 0; j < ans.ncols; j++) {
        ans.data[i][j] = data[j][i];
      }
    }
    return ans;
}

// Returns a matrix containing column means
Matrix Matrix::mean() {
  ld sum;
  Matrix ans(1, ncols);
  for (ll j = 0; j < ans.ncols; j++) {
  sum = 0;
    for (ll i = 0; i < nrows; i++) {
      sum += data[i][j];
    }
    ans.data[0][j] = sum/nrows;
  }
  return ans;
}

// Sets all elements to 1
Matrix Matrix::initializer() {
  Matrix ans(nrows, ncols);
  for (ll i = 0; i < nrows; i++) {
    for (ll j = 0; j < ncols; j++) {
      data[i][j]  = 1;
    }
  }
  return ans;
}

// Returns the determinant of the matrix
ld Matrix::determinant() {
  if (nrows != ncols) {
    cerr << "Dimension error (" << nrows << ", " << ncols << "): ";
    cerr << "Determinant can be computed only for a square matrix";
    exit(0);
  }
  vector<bool> row_mask(nrows, true), col_mask(nrows, true);
  return determinant(row_mask, col_mask);
}

// Returns the inverse of the matrix
Matrix Matrix::inverse() {
  vector<bool> row_mask(nrows, true), col_mask(nrows, true);
  ld det = this->determinant(row_mask, col_mask);
  if (nrows != ncols) {
    cerr << "Dimension error (" << nrows << ", " << ncols << "): ";
    cerr << "Inverse can be computed only for a square matrix";
    exit(0);
  } else if (det == 0) {
    cerr << "Error: Matrix is non-invertible (determinant is 0)\n";
    exit(0);
  }

  Matrix ans(nrows, nrows);
  for (ll i = 0; i < nrows; i++) {
    for (ll j = 0; j < ncols; j++) {
      // find cofactor of (j, i)
      row_mask[j] = false; col_mask[i] = false;
      ans.data[i][j] = ((i + j) % 2 ? -1 : 1) * determinant(row_mask, col_mask);
      row_mask[j] = true; col_mask[i] = true;
    }
  }
  return ans / det;
}

// Append a row before row with index 'before' and set all its elements
// to 'fill_val'
void Matrix::addRow(ll before, ll fill_val) {
  data.insert(data.begin() + before, vector<ld>(ncols, fill_val));
  ++nrows;
}

// Returns the number of rows
ll Matrix::getNumRows() const {
  return nrows;
}

// Returns the number of columns
ll Matrix::getNumCols() const {
  return ncols;
}

// Returns a matrix representing the sum of two matrices
Matrix operator+(const Matrix& m1, const Matrix& m2) {
  if (m1.nrows != m2.nrows || m1.ncols != m2.ncols) {
    cerr << "Dimension mismatch: ";
    cerr << "(" << m1.nrows << ", " << m1.ncols << ") + ";
    cerr << "(" << m2.nrows << ", " << m2.ncols << ")\n";
    exit(0);
  }
  Matrix ans(m1.nrows, m1.ncols);
  for (ll i = 0; i < ans.nrows; i++) {
    for (ll j = 0; j < ans.ncols; j++) {
      ans.data[i][j] = m1.data[i][j] + m2.data[i][j];
    }
  }
  return ans;
}

// Returns a matrix representing the difference of two matrices
Matrix operator-(const Matrix& m1, const Matrix& m2) {
  if (m1.nrows != m2.nrows || m1.ncols != m2.ncols) {
    cerr << "Dimension mismatch: ";
    cerr << "(" << m1.nrows << ", " << m1.ncols << ") + ";
    cerr << "(" << m2.nrows << ", " << m2.ncols << ")\n";
    exit(0);
  }
  Matrix ans(m1.nrows, m1.ncols);
  for (ll i = 0; i < ans.nrows; i++) {
    for (ll j = 0; j < ans.ncols; j++) {
      ans.data[i][j] = m1.data[i][j] - m2.data[i][j];
    }
  }
  return ans;
}

// Returns a matrix representing the product of two matrices
Matrix operator*(const Matrix& m1, const Matrix& m2) {
  if (m1.ncols != m2.nrows) {
    cerr << "Dimension mismatch: ";
    cerr << "(" << m1.nrows << ", " << m1.ncols << ") * ";
    cerr << "(" << m2.nrows << ", " << m2.ncols << ")\n";
    exit(0);
  }
  Matrix ans(m1.nrows, m2.ncols);
  for (ll i = 0; i < ans.nrows; i++) {
    for (ll j = 0; j < ans.ncols; j++) {
      ans.data[i][j] = 0;
      for (ll k = 0; k < m1.ncols; k++) {
        ans.data[i][j] += m1.data[i][k] * m2.data[k][j];
      }
    }
  }
  return ans;
}

// Applicable when 'm' is (1, 1) matrix. Returns the sum of the number
// and the matrix
ld operator+(const Matrix& m, ld num) {
  if (m.nrows != 1 || m.ncols != 1) {
    cerr << "Error: Attempting 'matrix' + 'number' when dimensions of matrix are (";
    cerr << m.nrows << ", " << m.ncols << ")\n";
    exit(0);
  }
  return m.data[0][0] + num;
}

// Applicable when 'm' is (1, 1) matrix. Returns the sum of the number
// and the matrix
ld operator+(ld num, const Matrix& m) {
  if (m.nrows != 1 || m.ncols != 1) {
    cerr << "Error: Attempting 'number' + 'matrix' when dimensions of matrix are (";
    cerr << m.nrows << ", " << m.ncols << ")\n";
    exit(0);
  }
  return num + m.data[0][0];
}

// Applicable when 'm' is (1, 1) matrix. Returns the difference of the number
// and the matrix
ld operator-(const Matrix& m, ld num) {
  if (m.nrows != 1 || m.ncols != 1) {
    cerr << "Error: Attempting 'matrix' - 'number' when dimensions of matrix are (";
    cerr << m.nrows << ", " << m.ncols << ")\n";
    exit(0);
  }
  return m.data[0][0] - num;
}

// Applicable when 'm' is (1, 1) matrix. Returns the difference of the number
// and the matrix
ld operator-(ld num, const Matrix& m) {
  if (m.nrows != 1 || m.ncols != 1) {
    cerr << "Error: Attempting 'number' - 'matrix' when dimensions of matrix are (";
    cerr << m.nrows << ", " << m.ncols << ")\n";
    exit(0);
  }
  return num - m.data[0][0];
}

// Applicable when 'm' is (1, 1) matrix. Returns the sum of the number
// and the matrix
ld& operator+=(ld& num, const Matrix& m) {
  if (m.nrows != 1 || m.ncols != 1) {
    cerr << "Error: Attempting 'number' += 'matrix' when dimensions of matrix are (";
    cerr << m.nrows << ", " << m.ncols << ")\n";
    exit(0);
  }
  num += m.data[0][0];
  return num;
}

// Applicable when 'm' is (1, 1) matrix. Returns the difference of the
// number and the matrix
ld& operator-=(ld& num, const Matrix& m) {
  if (m.nrows != 1 || m.ncols != 1) {
    cerr << "Error: Attempting 'number' -= 'matrix' when dimensions of matrix are (";
    cerr << m.nrows << ", " << m.ncols << ")\n";
    exit(0);
  }
  num -= m.data[0][0];
  return num;
}

// Applicable when 'm' is (1, 1) matrix. Returns the product of the
// number and the matrix
ld& operator*=(ld& num, const Matrix& m) {
  if (m.nrows != 1 || m.ncols != 1) {
    cerr << "Error: Attempting 'number' *= 'matrix' when dimensions of matrix are (";
    cerr << m.nrows << ", " << m.ncols << ")\n";
    exit(0);
  }
  num *= m.data[0][0];
  return num;
}

// Applicable when 'm' is (1, 1) matrix. Returns the division of the
// number and the matrix
ld& operator/=(ld& num, const Matrix& m) {
  if (m.nrows != 1 || m.ncols != 1) {
    cerr << "Error: Attempting 'number' /= 'matrix' when dimensions of matrix are (";
    cerr << m.nrows << ", " << m.ncols << ")\n";
    exit(0);
  }
  num /= m.data[0][0];
  return num;
}

// Returns a matrix which is the scalar multiplication of 'm' with 'num'
Matrix operator*(const Matrix& m, ld num) {
  Matrix ans(m);
  ans.element_wise_multiply(num);
  return ans;
}

// Returns a matrix which is the scalar multiplication of 'm' with 'num'
Matrix operator*(ld num, const Matrix& m) {
  Matrix ans(m);
  ans.element_wise_multiply(num);
  return ans;
}

// Returns a matrix which is the division of 'm' with 'num'
Matrix operator/(const Matrix& m, ld num) {
  Matrix ans(m);
  ans.element_wise_divide(num);
  return ans;
}

// Used to print the matrix
std::ostream& operator<<(std::ostream& out, const Matrix& m) {
  for (ll i = 0; i < m.nrows; i++) {
    for (ll j = 0; j < m.ncols; j++) {
      out << m.data[i][j] << " ";
    }
    out << "\n";
  }
  return out;
}

// Computes the elements of the submatrix defined by 'row_mask' and 
// 'col_mask'
ld Matrix::determinant(vector<bool>& row_mask, vector<bool>& col_mask) {
  ll r = 0, c = 0, nr = 0, nc = 0;
  for (size_t i = 0; i < row_mask.size(); i++) {
    if (row_mask[i]) {
      r = i;
      ++nr;
    } 
  }
  for (size_t i = 0; i < col_mask.size(); i++) {
    if (col_mask[i]) {
      c = i;
      ++nc;
    } 
  }

  if (nr == 0 || nc == 0) {
    return 1;
  } else if (nr == 1 && nc == 1) {
    return data[r][c];
  } else {
    ll r = 0;
    while (!row_mask[r++]);
    --r;
    row_mask[r] = false;

    int sign = 1;
    ld ans = 0;
    for (ll i = 0; i < ncols; i++) {
      if (col_mask[i]) {
        col_mask[i] = false;
        ans += sign * data[r][i] * determinant(row_mask, col_mask);
        sign *= -1;
        col_mask[i] = true;
      }
    }
    row_mask[r] = true;
    return ans;
  }
}

// Multiplies each element of the matrix with 'num'
void Matrix::element_wise_multiply(ld num) {
  for (ll i = 0; i < nrows; i++) {
    for (ll j = 0; j < ncols; j++) {
      data[i][j] *= num;
    }
  }
}

// Divides each element of the matrix with 'num'
void Matrix::element_wise_divide(ld num) {
  for (ll i = 0; i < nrows; i++) {
    for (ll j = 0; j < ncols; j++) {
      data[i][j] /= num;
    }
  }
}

// 'file' is the path to the file from where the dataset
// needs to be fetched
DataSet::DataSet(const string& file) {
  int isizecol = 4, row = 0, col;
  num_pos_examples = 0;
  num_neg_examples = 0;
  ld i;
  string line;
  ifstream myfile (file);
  if (myfile.is_open()) {
    while (getline (myfile,line)) {
      instances.push_back(Matrix(isizecol, 1));
      std::stringstream ss(line);
      col = 0;
      while (ss >> i) {
        if (col < isizecol) {
          instances[row][col][0] = i;
        } else {
          target_class.push_back(i);
          if (i) {
            ++num_pos_examples;
          } else {
            ++num_neg_examples;
          }
          break;
        }
        if (ss.peek() == ',' ) {
          ss.ignore();
          col++;
        }
      }
      row++;
    }
    myfile.close();
  } else {
    cerr << "Can't open file\n"; 
    exit(0);
  }
}

// Prints the dataset
void DataSet::print() {
  for (size_t i = 0; i < instances.size(); i++) {
    for (ll j = 0; j < instances[i].getNumRows(); j++) {
      cout << instances[i][j][0] << " ";
    }
    cout << target_class[i] << "\n";
  }
}

// 'tp', 'tn', 'fp', 'fn' represents the number of true positives,
// true negatives, false positives, false negatives respectively 
Stats::Stats(ll tp, ll tn, ll fp, ll fn) {
  this->tp = tp; this->tn = tn;
  this->fp = fp; this->fn = fn;
  f_measure = static_cast<ld>(2 * tp) / (2 * tp + fp + fn); 
  accuracy = static_cast<ld>(tp + tn) / (tp + tn + fp + fn);
  precision = static_cast<ld>(tp) / (tp + fp);
  recall = static_cast<ld>(tp) / (tp + fn);
}

// Prints the statistics
ostream& operator<<(ostream& out, const Stats& s) {
  out << "Accuracy: " << s.accuracy << "\n";
  out << "Recall: " << s.recall << "\n";
  out << "Precision: " << s.precision << "\n\n";
  // out << "F Measure: " << s.f_measure << "\n\n";
  out << "Confusion Matrix\n";
  out << s.tn << " " << s.fp << "\n" << s.fn << " " << s.tp << "\n";
  return out;
}
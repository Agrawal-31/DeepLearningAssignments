#ifndef CLASSIFIERS_H_
#define CLASSIFIERS_H_

#include <bits/stdc++.h>
#include "utilities.h"

using namespace std;

typedef long double ld;
typedef long long ll;

class Classifier {
public:
  // Tests a classifier on the dataset 'd'
  Stats test(const DataSet& d);
protected:
  // Classify the instance given by 'instance' using the classifier
  // This is used by 'test'
  virtual bool classify(const Matrix& instance) = 0;

  // Returns the sigmoid of 'x'
  ld sigmoid(ld x);
};

class DiscriminantClassifier: public Classifier {
public:
  DiscriminantClassifier();
  void run();
private:
  // Computes entropy
  long double loga(long double d);

  // Computes entropy
  ld entropy(ld noof1, ld isizerow, ld position, ld count);

  // Classify the instance given by 'instance' using the classifier
  // This is used by 'test'
  bool classify(const Matrix& instance);
  ll N;
}; 

class ProbabilisticGenerativeClassifier: public Classifier {
public:
  // Trains the classifier using the dataset 'd'
  void train(const DataSet& d);
private:
  // Classify the instance given by 'instance' using the classifier
  // This is used by 'test'
  bool classify(const Matrix& instance);
  Matrix w_transpose;
  ld w0;
};

class LogisticRegressionClassifier: public Classifier {
public:
  // Trains the classifier using the dataset 'd', where 'eta' is the learning
  // rate and 'epochs' is the number of iterations
  void train(const DataSet& d, ld eta, ld threshold, ll epochs);
private:
  // Error function
  ld error(const Matrix& w_transpose, const DataSet& d);

  // Classify the instance given by 'instance' using the classifier
  // This is used by 'test'
  bool classify(const Matrix& instance);

  // Appends an x0=1 to 'instance' and returns the new matrix 
  Matrix transform(const Matrix& instance);

  // Assigns random integers to the elements of matrix 'w_transpose'
  void initialize_weights();

  Matrix w_transpose;
};
#endif
#include <bits/stdc++.h>
#include "classifiers.h"
#include "utilities.h"

using namespace std;

typedef long double ld;
typedef long long ll;

// Tests a classifier on the dataset 'd'
Stats Classifier::test(const DataSet& d) {
  ll tp = 0, tn = 0, fp = 0, fn = 0;
  ll dsize = d.instances.size();
  for (ll i = 0; i < dsize; i++) {
    bool c = classify(d.instances[i]);

    // Compute the number of true positives, true negatives, false positives
    // and false negatives
    if (c == d.target_class[i]) {
      if (c) {
        ++tp;
      } else {
        ++tn;
      }
    } else {
      if (c) {
        ++fp;
      } else {
        ++fn;
      }
    }
  }
  Stats ans(tp, tn, fp, fn);
  return ans;
}

// Returns the sigmoid of 'x'
ld Classifier::sigmoid(ld x) {
  return 1 / (1 + exp(-x));
}

/*************************************************************************************/
DiscriminantClassifier::DiscriminantClassifier() {
  N = 4;
}

void DiscriminantClassifier::run() {
  long long isizerow = 960, isizecol = 4, row = 0, col = 0;
  long double i, j, k, key1, key2, max;
  int noof1 = 0, position = 0;
  Matrix a(isizerow, isizecol);
  //Reading Data From file and filling 
  //data to "a" matrix and pushing corresponding class to "gresult" vector.

  std::vector<long double> gresult;
  string line;
  ifstream myfile ("data/train.txt");
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
      std::stringstream ss(line);
      col = 0;
      while (ss >> i)
      {
          if(col < isizecol)
          {
            a[row][col] = i;
          }else
            {
              gresult.push_back(i);
              break;
            }

          if (ss.peek() == ',' )
          {
              ss.ignore();
              col++;
          }
      }
    row++;
    }
    myfile.close();
  }else cout << "Unable to open file"; 


  //Counting number of elements in each class
  int c1 = 0, c2;
  for(i = 0; i < isizerow; i++)
  {
    if(gresult.at(i) == 1)
    {
      c1++;
    }
  }
  c2 = isizerow - c1;

  Matrix b(c1, isizecol), c(isizerow - c1, isizecol);

  //Differentiating both classes to different matrixes "b"1 class & "c" - 0 class
  int rowb = 0, rowc = 0;
  for (i=0; i< isizerow; i++)
  {
    if(gresult.at(i) == 0)
    {
      for (j=0; j< isizecol; j++)
      {
        c[rowc][j] = a[i][j]; 
      }
      rowc++;
    }else
    {
      for (j=0; j< isizecol; j++)
      {
        b[rowb][j] = a[i][j]; 
      }
      rowb++;
    }
  }

  Matrix sw(isizecol,isizecol);
  Matrix identityb(c1, c1);
  Matrix identityc(c2, c2);

  //identity matrices for further calculations

  identityb.initializer();
  identityc.initializer();

  Matrix tempb(c1, isizecol);
  Matrix tempc(c2, isizecol);
  tempb = (b - ((identityb*b)/c1)); // x - m
  tempc = (c - ((identityc*c)/c2)); // x - m

  //Calculating within class co-varience and adding them
  sw =  ((tempb.transpose())*tempb) + ((tempc.transpose())*tempc); //(y- mu)(y - mu)T

  Matrix inversesw(isizecol, isizecol);

  inversesw = sw.inverse(); 

  Matrix finalcorrec(1,4);

  //Final Projection vector "finalcorrec" is calculated

  finalcorrec = (inversesw*((b.mean() - c.mean()).transpose())); // J(W) = Tr{sw-1*sb}

  Matrix temp(1, isizecol), projectedvalue(1,1), conclu(isizerow, 2);

  //filling projected values(Sw * training vectors) and there 
  //corresponding classes to conclu - [rojected value,actual class]
  for(i = 0; i < isizerow; i++)
  {
      for(j = 0; j < 4; j++)
      {
        temp[0][j] = a[i][j];
      }
      projectedvalue =  temp*finalcorrec;

      conclu[i][0] = projectedvalue[0][0];
      conclu[i][1] = gresult.at(i);     
  }

  //Sorting projected vectors with respect to projected 1-D value 
  //with corresponding class information also

  for (i = 1; i < isizerow; i++)
  {
      key1 = conclu[i][0];
      key2 = conclu[i][1];
      j = i-1;

     while (j >= 0 && conclu[j][0] > key1)
     {
         conclu[j+1][0] = conclu[j][0];
         conclu[j+1][1] = conclu[j][1];
         j = j-1;
     }
     conclu[j+1][0] = key1;
     conclu[j+1][1] = key2;
  }

  std::vector<long double> threshold;

  //find mean of all consecutive projected values for entropy calculation
  for(i = 0; i < isizerow-1; i++)
  {
    threshold.push_back((conclu[i][0] + conclu[i+1][0])/2);
  }

  //Finding total no of members of each class
  for(i = 0; i < isizerow; i++)
  {
    if(conclu[i][1] == 1)
    {
      noof1++;
    }
  }

  std::vector<long double> entropyval;

  //Pushing entropy Values to "entropyval" vector
  for(i = 0; i < isizerow-1; i++)
  {
    j = 0;
    while(conclu[j][0] < threshold[i])
    {
      j++;
    }

    int count = 0;

    for(k = 0; k < j; k++)
    {
      if(conclu[k][1] == 1)
      {
        count++;
      }
    }
    entropyval.push_back(entropy(noof1, isizerow, j,count));
  }

  //Finding Maximum Entropy and corresponding threshold value
  max = entropyval[0];
  for(i = 0; i < isizerow-1; i++)
  {
    if(max < entropyval.at(i))
    {
      max = entropyval.at(i);
      position = i;
    }
  }
  // cout <<"Max Entropy" <<  max << endl;
  max = 0;
  ld finalthresh = threshold.at(position);

  // cout << "threshhold: " <<finalthresh<< endl;

  //Calculating total no of elements in predicted classes
  for(i = 0; i < isizerow; i++)
  {
    if(conclu[i][0] < finalthresh)
    {
      max++;
    }
  }

  //***TESTING STARTS***
  //Reading data and filling to"d" matrix
  isizerow = 412; row = 0; col = 0; isizecol = 4; 
  std::vector<ld> testans;;
  Matrix d(isizerow, isizecol);
  ifstream newfile ("data/test.txt");
  if (newfile.is_open())
  {
    while ( getline (newfile,line) )
    {
      std::stringstream ss(line);
      col = 0;
      while (ss >> i)
      {
          if(col < isizecol)
          {
            d[row][col] = i;
          }else
            {
              testans.push_back(i);
              break;
            }

          if (ss.peek() == ',' )
          {
              ss.ignore();
              col++;
          }
      }
    row++;
    }
  newfile.close();
  }else {cout << "Unable to open file";exit(0);}

  //Finding total no elements in each class
  int good = 0;
  for(i = 0; i < 412; i++)
  {
    if (testans[i] == 1)
    {
      good++;      }
  }

  Matrix conclufin(isizerow, 2);

  //Filling projected Vlaues with corresponding classes to a "conclufin"
  //matrix
  for(i = 0; i < isizerow; i++)
  {
      for(j = 0; j < 4; j++)
      {
        temp[0][j] = d[i][j];
      }
      projectedvalue =  temp*finalcorrec;

      conclufin[i][0] = projectedvalue[0][0];
      conclufin[i][1] = testans.at(i);     
  }

  //Finding numbers of correctly classified and misclassified

  ld a1 = 0, b0 = 0, a0 = 0, b1 = 0;
  for(i = 0; i < isizerow; i++)
  {
    if(conclufin[i][0] < finalthresh && conclufin[i][1] != 0)
    {
      a1++;
    }else if(conclufin[i][0] < finalthresh){
      a0++;
    }

    if(conclufin[i][0] > finalthresh && conclufin[i][1] != 1  )
    {
      b0++;
    }else if(conclufin[i][0] > finalthresh){
      b1++;
    }
  }


  ld accuracy = (a0 + b1)/412;
  ld recall = (b1)/good;
  ld precision = b1/(b0 + b1);

  cout << "Accuracy: " << accuracy<<endl;
  cout << "Recall: "<< recall <<endl;
  cout << "Precision: " << precision << endl << endl;
  
  cout << "Confusion Matrix"<< endl;

  cout << a0 <<"  "<<b0 <<endl;
  cout << a1 << " "<< b1 << endl;
}

bool DiscriminantClassifier::classify(const Matrix& instance) {
  return true;
}

long double DiscriminantClassifier::loga(long double d)
{
    if(d == 0)
      return 0;
    return log(d);
}

ld DiscriminantClassifier::entropy(ld noof1, ld isizerow, ld position, ld count)
{
  ld a, b, c, d;
  a = count/position;
  b = 1-a;
  c = (noof1-count)/(isizerow-position);
  d = (1-c);
  return (a * loga(a) + b*loga(b) + c * loga(c) + d*loga(d));
}

/*************************************************************************************/

// Trains the classifier using the dataset 'd'. Computes 'w_transpose' and
// 'w0'
void ProbabilisticGenerativeClassifier::train(const DataSet& d) {
  // Compute mu1 and mu2
  ll n = d.instances.size(), num_features = d.instances[0].getNumRows();
  ll n1 = d.num_pos_examples, n2 = d.num_neg_examples;
  
  // Compute mu1 and mu2
  Matrix mu1(num_features, 1), mu2(num_features, 1);
  mu1 = {}; mu2 = {};
  for (ll i = 0; i < n; i++) {
    mu1 += d.target_class[i] * d.instances[i]; //mu1 = tn*xn
    mu2 += (1 - d.target_class[i]) * d.instances[i]; //mu1 = (1-tn)*xn
  }
  mu1 /= n1; mu2 /= n2;

  // Compute s
  Matrix s(num_features, num_features); // S = sigma((xn - mu1)*(xn - mu1)T)
  s = {};
  for (ll i = 0; i < n; i++) {
    if (d.target_class[i]) {
      Matrix temp = d.instances[i] - mu1;
      s += temp * temp.transpose();
    } else {
      Matrix temp = d.instances[i] - mu2;
      s += temp * temp.transpose();
    }
  }
  s /= n;
  Matrix s_inv = s.inverse();

  // Compute w_transpose and w0
  w_transpose = (s_inv * (mu1 - mu2)).transpose();
  w0 = 0;
  w0 += -0.5 * mu1.transpose() * s_inv * mu1; 
  w0 += 0.5 * mu2.transpose() * s_inv * mu2; 
  w0 += log((static_cast<ld>(n1) / n) / (static_cast<ld>(n2) / n));
}

// Classify the instance given by 'instance' using the classifier
// This is used by 'test'
bool ProbabilisticGenerativeClassifier::classify(const Matrix& instance) {
  return sigmoid(w_transpose * instance + w0) >= 0.5; //greater than or equal 0.5 then class 1
}

/*************************************************************************************/

// Trains the classifier using the dataset 'd', where 'eta' is the learning
// rate and 'epochs' is the number of iterations
// phi is x itself (identity function)
void LogisticRegressionClassifier::train(const DataSet& d, ld eta, ld threshold, ll epochs) {
  bool use_epochs = (threshold < 0);
  ll dsize = d.instances.size(), num_features = d.instances[0].getNumRows();
  Matrix w(num_features + 1, 1), grad(num_features + 1, 1);
  initialize_weights();

  vector<Matrix> transf_inst(dsize);
  for (ll i = 0; i < dsize; i++) {
    transf_inst[i] = transform(d.instances[i]);
  }

  if (use_epochs) { //stop at given number of epochs
    while (epochs--) {
      grad = {};
      Matrix wt = w.transpose();
      for (ll i = 0; i < dsize; i++) {
        grad += (sigmoid(wt * transf_inst[i] + 0) - d.target_class[i]) * transf_inst[i]; //delta error = (yn - tn)phi
      }
      w = w - (eta * grad);
    } 
  } else { //stop once the the error change is less than a threshold value
    ld err, new_err;
    err = error(w.transpose(), d);
    new_err = err + threshold + 1;
    while (fabs(new_err - err) > threshold) {
      err = new_err;
      grad = {};
      Matrix wt = w.transpose();
      for (ll i = 0; i < dsize; i++) {
        grad += (sigmoid(wt * transf_inst[i] + 0) - d.target_class[i]) * transf_inst[i];
      }
      w -= eta * grad;
      new_err = error(w.transpose(), d);
    }
  }

  w_transpose = w.transpose();
}

// ECross entropy error function
ld LogisticRegressionClassifier::error(const Matrix& w_transpose, const DataSet& d) {
  ll dsize = d.instances.size();
   vector<Matrix> transf_inst(dsize);
  for (ll i = 0; i < dsize; i++) {
    transf_inst[i] = transform(d.instances[i]);
  }
  
  ld ans = 0;
  for (ll i = 0; i < dsize; i++) {
    ld yn = sigmoid(w_transpose * transf_inst[i] + 0);
    ans += d.target_class[i] ? log(yn) : log(1 - yn); //Error function = tn*ln(yn) + (1-tn)*ln(1-yn)
  }
  return -ans;
}

// Classify the instance given by 'instance' using the classifier
// This is used by 'test'
bool LogisticRegressionClassifier::classify(const Matrix& instance) {
  return sigmoid(w_transpose * transform(instance) + 0) >= 0.5;
}

// Appends an x0=1 to 'instance' and returns the new matrix 
Matrix LogisticRegressionClassifier::transform(const Matrix& instance) {
  Matrix ans = instance;
  ans.addRow(0, 1);
  return ans;
}

// Assigns random integers to the elements of matrix 'w_transpose'
void LogisticRegressionClassifier::initialize_weights() {
  // Assumes 'w_transpose' is a column matrix
  ll nrows = w_transpose.getNumRows();
  default_random_engine gen;
  normal_distribution<ld> dist(0, 1 / sqrt(nrows - 1));
  for (ll i = 0; i < nrows; i++) {
      w_transpose[i][0] = dist(gen);
  }
}
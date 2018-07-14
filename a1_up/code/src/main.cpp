#include <bits/stdc++.h>
#include "classifiers.h"
#include "utilities.h"

using namespace std;

typedef long long ll;
typedef long double ld;

int main() {
  // Load the datasets
  DataSet train_data("data/train.txt");
  DataSet test_data("data/test.txt");

  cout << "Fisher Discriminant Classifier\n";
  cout << "`````````````````````````````\n";
  DiscriminantClassifier dc;
  dc.run();
  cout << "\n";

  cout << "Probabilistic Generative Classifier\n";
  cout << "``````````````````````````````````\n";
  ProbabilisticGenerativeClassifier pgc;
  pgc.train(train_data);
  Stats pgc_stats = pgc.test(test_data);
  cout << pgc_stats << "\n";

  cout << "Logistic Regression Classifier\n";
  cout << "`````````````````````````````\n";
  LogisticRegressionClassifier lrc;
  lrc.train(train_data, 0.0015, -1, 15000);
  // lrc.train(train_data, 0.01, 1e-5, -1);
  Stats lrc_stats = lrc.test(test_data);
  cout << lrc_stats;

  return 0;
}
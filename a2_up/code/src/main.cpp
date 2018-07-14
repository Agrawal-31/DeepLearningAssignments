#include <bits/stdc++.h>
#include "utilities.h"
#include "neural_net.h"

using namespace std;

int main() {
  DataSet train_data("data/train.txt");
  DataSet test_data("data/test.txt");
  DataSet validation_data("data/validation.txt");

  NeuralNet nn({64, 10, 10});
  nn.train(train_data, validation_data, test_data, false, 100, 3000);
  
  cout << "Final Accuracy: " << nn.test(test_data) << endl;
  return 0;
}
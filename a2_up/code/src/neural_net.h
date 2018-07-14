#ifndef NEURAL_NET_H_
#define NEURAL_NET_H_

#include <bits/stdc++.h>
#include "utilities.h"

using namespace std;

class NeuralNet {
public:
  // Argument is an initializer list consisting of sizes of each of the
  // layers in the network
  NeuralNet(const initializer_list<ll>& lsizes);

  // Trains the network
  void train(const DataSet& train_data, const DataSet& validation_data,
    const DataSet& test_data, bool auto_stop = false, ll batch_size = 100, 
    ll epochs = 3000, ld momentum = 0.9, ld eta = 2, ld eps = 1e-8);
  
  // Returns the accuracy of the network on test_data
  ld test(const DataSet& test_data);

private:
  // Classifes a given instance by feedforwarding
  ll classify(const Matrix& instance);

  // Compute the weighted inputs by feed forwarding
  vector<Matrix> compute_weighted_inputs(
    const Matrix& instance, bool only_output = false);

  // Compute the errors using backpropagation
  vector<Matrix> compute_errors(vector<Matrix>& e,
    short target_class);

  // Returns the logistic sigmoid of x
  ld sigmoid(ld x);

  // Returns the value of sigmoid derivative applied on x
  ld sigmoid_prime(ld x);

  // Vectorized form of sigmoid_prime
  Matrix sigmoid_prime(Matrix& m);

  // Activation function
  Matrix act_func(Matrix& m);

  // Randomly initializes the weights and biases with -1 to 1
  void randInitParams();

  // Layers are numbered from 0 to nlayers - 1

  // w[i][j][k] is the weight of the edge from node k in layer i - 1 to node
  // j in layer i
  // w[i] is of size (lsizes[i], lsizes[i - 1])
  // w is of size (nlayers - 1)
  vector<Matrix> w;

  // lsizes[i] is the number of neurons in layer i
  vector<ll> lsizes;

  // b[i][j][0] is the bias of the neuron j in layer i
  // b[i] is of size (lsizes[i], 1)
  // Size of b is (nlayers - 1)
  vector<Matrix> b;

  // Number of layers
  ll nlayers;
};

#endif
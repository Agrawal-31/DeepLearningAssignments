#include <bits/stdc++.h>
#include "utilities.h"
#include "neural_net.h"

// Argument is an initializer list consisting of sizes of each of the
// layers in the network
NeuralNet::NeuralNet(const initializer_list<ll>& lsizes) {
  this->lsizes = lsizes;
  this->nlayers = lsizes.size();
  w.resize(nlayers);
  b.resize(nlayers);
  for (ll i = 1; i < nlayers; i++) {
    // For layer i
    w[i] = Matrix(this->lsizes[i], this->lsizes[i - 1]);
    b[i] = Matrix(this->lsizes[i], 1);
  }
}

// Trains the network
void NeuralNet::train(const DataSet& train_data,
  const DataSet& validation_data, const DataSet& test_data, bool auto_stop, ll batch_size, ll epochs,
  ld momentum, ld eta, ld eps) {
  ll dsize = train_data.instances.size();
  randInitParams();

  // del_c_w[1..nlayers - 1]
  // del_c_b[i][j][k] is the derivative of c wrt w, where w is the weight of 
  // the edge connecting neuron k of layer i - 1 to neuron j of layer i
  //
  // del_c_b[1..nlayers - 1]
  // del_c_b[i][j][0] is the derivative of c wrt b, where b is the bias of
  // neuron j of layer i
  vector<Matrix> del_c_w(nlayers), del_c_b(nlayers);
  // For momentum
  vector<Matrix> v_w(nlayers), v_b(nlayers);
  // For AdaGrad
  vector<Matrix> running_del_c_w(nlayers), running_del_c_b(nlayers);

  // Initialization of matrices and setting them to the proper size
  for (ll i = 1; i < nlayers; i++) {
    // For layer i
    del_c_w[i] = Matrix(lsizes[i], lsizes[i - 1]);
    del_c_b[i] = Matrix(lsizes[i], 1);
    
    v_w[i] = Matrix(lsizes[i], lsizes[i - 1]);
    v_b[i] = Matrix(lsizes[i], 1);
    
    v_w[i] = {0};
    v_b[i] = {0};

    running_del_c_w[i] = Matrix(lsizes[i], lsizes[i - 1]);
    running_del_c_w[i] = {eps};
    running_del_c_b[i] = Matrix(lsizes[i], 1);
    running_del_c_b[i] = {eps};
  }

  bool cont = true;
  ld curr, prev = 0;
  for (ll epoch = 0; epoch < epochs && cont; epoch++) {
    cout << epoch << ",";   
    // Mini batch gradient descent
    for (ll i = 0; i < dsize; i += batch_size) {
      for (ll j = 1; j < nlayers; j++) {
        del_c_w[j] = {0};
        del_c_b[j] = {0};
      }
      ll lo = i, hi = min(i + batch_size - 1, dsize - 1);
      
      // Now the minibatch to consider is train_data[lo..hi]
      for (ll j = lo; j <= hi; j++) {
        // z[0..nlayers - 1]
        // z[i][j][0] is the weighted input to neuron j in layer i
        vector<Matrix> z = compute_weighted_inputs(train_data.instances[j]);

        // e[1..nlayers - 1]
        // e[i][j][0] is the error of the neuron j in layer i
        vector<Matrix> e = compute_errors(z, train_data.target_class[j]);

        // Combine del_c_w, del_c_b for this example with that of previous
        // ones. This is essentially to compute sigma(del_c_w/b) over all
        // examples in the minibatch
        for (ll k = 1; k < nlayers; k++) {
          del_c_w[k] += e[k] * act_func(z[k - 1]).transpose();
          del_c_b[k] += e[k];
        }
      }

      
      // Running sum of squares of gradients required for AdaGrad
      for (ll j = 1; j < nlayers; j++) {
        running_del_c_w[j] += 
          del_c_w[j].apply([](ld x) -> ld {return x * x;}); 
        running_del_c_b[j] += 
          del_c_b[j].apply([](ld x) -> ld {return x * x;});
      }
      
      // Gradient descent
      for (ll j = 1; j < nlayers; j++) {
        v_w[j] = (momentum * v_w[j]) + 
          (eta * hadamard_prod(
          running_del_c_w[j].apply([](ld x) -> ld {return 1 / sqrt(x);}), 
          (del_c_w[j] / (hi - lo + 1))));

        v_b[j] = (momentum * v_b[j]) + 
          (eta * hadamard_prod(
          running_del_c_b[j].apply([](ld x) -> ld {return 1 / sqrt(x);}), 
          (del_c_b[j] / (hi - lo + 1))));

        w[j] -= v_w[j];
        b[j] -= v_b[j];
      }
    }
    cout << test(validation_data) << endl;
    // cout << test(validation_data) << "," << test(train_data) << "\n";
    if (auto_stop) {
      curr = test(validation_data);
      cout << curr << endl;
      if (curr - prev < -0.5) {
        cont = false;
      } else {
        prev = curr;
      }
    }
  }
}

// Returns the accuracy of the network on test_data
ld NeuralNet::test(const DataSet& test_data) {
  ll dsize = test_data.instances.size();
  ll t = 0, f = 0;
  for (ll i = 0; i < dsize; i++) {
    if (classify(test_data.instances[i]) == test_data.target_class[i]) {
      ++t;
    } else {
      ++f;
    }
  }
  return static_cast<ld>(t * 100) / (t + f);
}

// Classifes a given instance by feedforwarding
ll NeuralNet::classify(const Matrix& instance) {
  vector<Matrix> output;
  output = compute_weighted_inputs(instance, true);
  output[0] = act_func(output[0]);
  ll out_size = lsizes[nlayers - 1], max_idx = 0;
  for (ll i = 0; i < out_size; i++) {
    if (output[0][i][0] > output[0][max_idx][0]) {
      max_idx = i;
    }
  }
  return max_idx;
}

// Compute the weighted inputs by feed forwarding
vector<Matrix> NeuralNet::compute_weighted_inputs(
  const Matrix& instance, bool only_output) {
  
  // Assume instance to be a column matrix
  vector<Matrix> z;
  
  // For layer 0
  Matrix z_layer = instance;
  if (!only_output) {
    z.push_back(z_layer);
  }

  // For layer 1
  z_layer = w[1] * z_layer + b[1];
  if (!only_output) {
    z.push_back(z_layer);
  }
  
  // For other layers
  for (ll i = 2; i < nlayers; i++) {
    // Compute weighted input for layer i
    z_layer = w[i] * act_func(z_layer) + b[i];
    if (!only_output || i == nlayers - 1) {
      z.push_back(z_layer);
    }
  }
  return z;
}

// Compute the errors using backpropagation
vector<Matrix> NeuralNet::compute_errors(vector<Matrix>& z,
  short target_class) {
  vector<Matrix> e(nlayers);

  // Fill output layer errors
  Matrix target_vect(lsizes[nlayers - 1], 1);
  target_vect = {0};
  target_vect[target_class][0] = 1;
  e[nlayers - 1] = act_func(z[nlayers - 1]) - target_vect;

  // Backpropagate the errors
  for (ll i = nlayers - 2; i > 0; i--) {
    e[i] = hadamard_prod(w[i + 1].transpose() * e[i + 1], sigmoid_prime(z[i]));
  }
  return e;
}

// Returns the logistic sigmoid of x
ld NeuralNet::sigmoid(ld x) {
  return exp(x) / (1 + exp(x));
}

// Returns the value of sigmoid derivative applied on x
ld NeuralNet::sigmoid_prime(ld x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

// Vectorized form of sigmoid_prime
Matrix NeuralNet::sigmoid_prime(Matrix& m) {
  ll r = m.getNumRows(), c = m.getNumCols();
  Matrix res(r, c);
  for (ll i = 0; i < r; i++) {
    for (ll j = 0; j < c; j++) {
      res[i][j] = sigmoid_prime(m[i][j]);
    }
  }
  return res;
}

// Activation function
Matrix NeuralNet::act_func(Matrix& m) {
  Matrix res = m;
  ll c = m.getNumCols(), r = m.getNumRows();
  for (ll i = 0; i < r; i++) {
    for (ll j = 0; j < c; j++) {
      res[i][j] = sigmoid(m[i][j]);
    }
  }
  return res;
}

// Randomly initializes the weights and biases with -1 to 1
void NeuralNet::randInitParams() {
  default_random_engine gen;
  uniform_real_distribution<double> dist(-1.0, 1.0);
  for (ll i = 1; i < nlayers; i++) {
    for (ll j = 0; j < lsizes[i]; j++) {
      for (ll k = 0; k < lsizes[i - 1]; k++) {
        w[i][j][k] = dist(gen);
      }
    }
  }
  for (ll i = 1; i < nlayers; i++) {
    for (ll j = 0; j < lsizes[i]; j++) {
      b[i][j][0] = dist(gen);
    }
  }
}
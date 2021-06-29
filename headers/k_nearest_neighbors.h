#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

namespace SimpleML
{
	class KNN
	{
	private:
		int K;
		int n_class;
		MatrixXf features;
		VectorXi labels;
	public:
		KNN(int K);
		void fit(const MatrixXf& X, const VectorXi& Y);
		VectorXi predict(const MatrixXf& X);
	private:
		vector<float> calculate_euclidean_norms(const RowVectorXf& xt);
		vector<int> select_K_neighbors(const vector<float>& norms);
		int select_most_frequent(const vector<int>& frequencies);
	};

	KNN::KNN(int K) : K(K), n_class(0)
	{
		if (K <= 0) {
			cout << "Error(KNN(int): Invalid neighbor size." << endl;
			exit(1);
		}
	}

	void KNN::fit(const MatrixXf& X, const VectorXi& Y)
	{
		features = X;
		labels = Y;
		n_class = *std::max_element(Y.data(), Y.data() + Y.size()) + 1;
	}

	VectorXi KNN::predict(const MatrixXf& X)
	{
		VectorXi predicted(X.rows());
		for (int i = 0; i < X.rows(); i++) {
			vector<float> norms = calculate_euclidean_norms(X.row(i));
			vector<int> freqs = select_K_neighbors(norms);
			predicted[i] = select_most_frequent(freqs);
		}
		return predicted;
	}

	vector<float> KNN::calculate_euclidean_norms(const RowVectorXf& xt)
	{
		vector<float> norms(features.rows());
		for (int i = 0; i < features.rows(); i++) {
			norms[i] = (features.row(i) - xt).norm();
		}
		return norms;
	}

	vector<int> KNN::select_K_neighbors(const vector<float>& norms)
	{
		// argsort
		vector<int> indicies(norms.size());
		std::iota(indicies.begin(), indicies.end(), 0);
		std::stable_sort(indicies.begin(), indicies.end(),
			[&](int i, int j) { return norms[i] < norms[j]; });

		// select K neighbors
		vector<int> neighbors(n_class);
		for (int i = 0; i < K; i++) {
			int label = labels[indicies[i]];
			neighbors[label]++;
		}

		return neighbors;
	}

	int KNN::select_most_frequent(const vector<int>& frequencies)
	{
		return (int)std::distance(frequencies.begin(),
			std::max_element(frequencies.begin(), frequencies.end()));
	}
}
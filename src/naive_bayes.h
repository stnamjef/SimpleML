#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

namespace SimpleML
{
	class NaiveBayes
	{
	private:
		int n_class;
		RowVectorXf* mu;
		MatrixXf* sigma;
		RowVectorXf prior;
	public:
		NaiveBayes();
		~NaiveBayes();
		void fit(const MatrixXf& X, const VectorXi& Y);
		VectorXi predict(const MatrixXf& X);
	private:
		vector<MatrixXf> split_by_class(const MatrixXf& X, const VectorXi& Y);
	};

	NaiveBayes::NaiveBayes() : n_class(0), mu(nullptr), sigma(nullptr) {}

	NaiveBayes::~NaiveBayes()
	{
		delete[] mu;
		delete[] sigma;
	}

	void NaiveBayes::fit(const MatrixXf& X, const VectorXi& Y)
	{
		n_class = *std::max_element(Y.data(), Y.data() + Y.size()) + 1;
		
		mu = new RowVectorXf[n_class];
		sigma = new MatrixXf[n_class];
		prior.resize(n_class);

		// split data by class
		vector<MatrixXf> splits = split_by_class(X, Y);

		// calculate prior probability of each class
		for (int i = 0; i < n_class; i++) {
			prior[i] = (float)splits[i].rows() / X.rows();
		}

		// calculate mu & sigma of each class
		for (int i = 0; i < n_class; i++) {
			mu[i] = splits[i].colwise().mean();
			MatrixXf centered = (splits[i].rowwise() - mu[i]);
			sigma[i] = centered.transpose() * centered / (float)(splits[i].size() - 1);
		}
	}

	vector<MatrixXf> NaiveBayes::split_by_class(const MatrixXf& X, const VectorXi& Y)
	{
		vector<vector<int>> indicies(n_class);
		for (int i = 0; i < Y.size(); i++) {
			indicies[Y[i]].push_back(i);
		}
		
		vector<MatrixXf> splits(n_class);
		for (int i = 0; i < indicies.size(); i++) {
			MatrixXf split(indicies[i].size(), X.cols());
			for (int j = 0; j < indicies[i].size(); j++) {
				split.row(j) = X.row(indicies[i][j]);
			}
			splits[i] = split;
		}

		return splits;
	}

	VectorXi NaiveBayes::predict(const MatrixXf& X)
	{
		VectorXi predicted(X.rows());
		for (int i = 0; i < X.rows(); i++) {
			vector<float> prob(n_class);
			for (int j = 0; j < n_class; j++) {
				prob[j] = multivariate_normal(X.row(i), mu[j], sigma[j]);
				prob[j] *= prior[j];
			}
			predicted[i] = (int)std::distance(prob.begin(), std::max_element(prob.begin(), prob.end()));
		}
		return predicted;
	}
}
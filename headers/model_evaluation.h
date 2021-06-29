#pragma once
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <limits>
#include <Eigen/Dense>
#include "common.h"
using namespace std;
using namespace Eigen;

namespace SimpleML
{
	template<class T>
	float evaluate_classification_model(T& model, const MatrixXf& X, const VectorXi& Y, int n_fold);

	vector<vector<int>> split_to_folds(const MatrixXf& X, int n_fold);

	MatrixXf train_feature(const MatrixXf& X, const vector<vector<int>>& folds, int except);

	VectorXi train_label(const VectorXi& Y, const vector<vector<int>>& folds, int except);

	MatrixXf test_feature(const MatrixXf& X, const vector<vector<int>>& folds, int include);

	VectorXi test_label(const VectorXi& Y, const vector<vector<int>>& folds, int include);

	float calc_accuracy(const VectorXi& actual, const VectorXi& predicted);

	template<class T>
	float evaluate_clustering_model(T& model, const MatrixXf& X);

	float silhouette_score(const MatrixXf& X, const vector<vector<int>>& clusters);

	float mean_distance(const RowVectorXf& p, const MatrixXf& X, const vector<int>& cluster);

	template<class T>
	float evaluate_classification_model(T& model, const MatrixXf& X, const VectorXi& Y, int n_fold)
	{
		vector<vector<int>> folds = split_to_folds(X, n_fold);

		float accuracy = 0;
		for (int i = 0; i < n_fold; i++) {
			model.fit(train_feature(X, folds, i), train_label(Y, folds, i));

			VectorXi predicted = model.predict(test_feature(X, folds, i));
			accuracy += calc_accuracy(test_label(Y, folds, i), predicted);
		}
		return accuracy / n_fold;
	}

	vector<vector<int>> split_to_folds(const MatrixXf& X, int n_fold)
	{
		vector<int> rand_num(X.rows());
		std::iota(rand_num.begin(), rand_num.end(), 0);
		unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(rand_num.begin(), rand_num.end(), std::default_random_engine(seed));

		int fold_size = (int)X.rows() / n_fold;
		vector<vector<int>> folds(n_fold, vector<int>(fold_size));
		for (int i = 0; i < n_fold; i++) {
			for (int j = 0; j < fold_size; j++) {
				folds[i][j] = rand_num[j + fold_size * i];
			}
		}
		return folds;
	}

	MatrixXf train_feature(const MatrixXf& X, const vector<vector<int>>& folds, int except)
	{
		size_t row = (folds.size() - 1) * folds[0].size();
		MatrixXf X_fold((Eigen::Index)row, X.cols());

		int i = 0;
		for (int j = 0; j < folds.size(); j++) {
			if (j == except)
				continue;

			for (const int& idx : folds[j]) {
				X_fold.row(i) = X.row(idx);
				i++;
			}
		}
		return X_fold;
	}

	VectorXi train_label(const VectorXi& Y, const vector<vector<int>>& folds, int except)
	{
		size_t size = (folds.size() - 1) * folds[0].size();
		VectorXi Y_fold((Eigen::Index)size);

		int i = 0;
		for (int j = 0; j < folds.size(); j++) {
			if (j == except)
				continue;

			for (const int& idx : folds[j]) {
				Y_fold[i] = Y[idx];
				i++;
			}
		}
		return Y_fold;
	}

	MatrixXf test_feature(const MatrixXf& X, const vector<vector<int>>& folds, int include)
	{
		MatrixXf X_fold((Eigen::Index)folds[0].size(), X.cols());

		int i = 0;
		for (const int& idx : folds[include]) {
			X_fold.row(i) = X.row(idx);
			i++;
		}
		return X_fold;
	}

	VectorXi test_label(const VectorXi& Y, const vector<vector<int>>& folds, int include)
	{
		VectorXi Y_fold((Eigen::Index)folds[0].size());

		int i = 0;
		for (const int& idx : folds[include]) {
			Y_fold[i] = Y[idx];
			i++;
		}
		return Y_fold;
	}

	float calc_accuracy(const VectorXi& actual, const VectorXi& predicted)
	{
		float correct = 0;
		for (int i = 0; i < actual.size(); i++) {
			if (actual[i] == predicted[i]) {
				correct++;
			}
		}
		return correct / actual.size();
	}

	template<class T>
	float evaluate_clustering_model(T& model, const MatrixXf& X)
	{
		model.fit(X);
		vector<vector<int>> clusters = model.predict(X);
		return silhouette_score(X, clusters);
	}

	float silhouette_score(const MatrixXf& X, const vector<vector<int>>& clusters)
	{
		float score = 0;
		for (int i = 0; i < clusters.size(); i++) {
			for (int j = 0; j < clusters[i].size(); j++) {
				int idx = clusters[i][j];
				float a = -1, b = -1;
				for (int k = 0; k < clusters.size(); k++) {
					float temp = mean_distance(X.row(idx), X, clusters[k]);
					if (i == k)
						a = temp;
					else if ((b == -1) || (b > temp))
						b = temp;
				}
				score += (b - a) / std::max(a, b);
			}
		}
		return score / (float)X.rows();
	}

	float mean_distance(const RowVectorXf& p, const MatrixXf& X, const vector<int>& cluster)
	{
		float dist = 0;
		for (const int& i : cluster) {
			dist += euclidean_norm(p, X.row(i));
		}
		return dist / (float)cluster.size();
	}
}
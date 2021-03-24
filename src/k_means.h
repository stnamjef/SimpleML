#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>
#include "common.h"
using namespace std;
using namespace Eigen;

namespace SimpleML
{
	class KMeans
	{
	private:
		int K;
		RowVectorXf* centers;
	public:
		KMeans(int K);
		~KMeans();
		void fit(const MatrixXf& X, string init = "kmpp");
		vector<vector<int>> predict(const MatrixXf& X);
		RowVectorXf* get_centers() const;
	private:
		void kmpp_init_center(const MatrixXf& X);
		void rand_init_center(const MatrixXf& X);
		int nearest_center(const RowVectorXf& x, int n_center_initialized);
		vector<vector<int>> make_clusters(const MatrixXf& X);
		void update_centers(const MatrixXf& X, const vector<vector<int>>& clusters);
	};

	KMeans::KMeans(int K) : K(K) { centers = new RowVectorXf[K]; }

	KMeans::~KMeans() { delete[] centers; }

	void KMeans::fit(const MatrixXf& X, string init)
	{
		if (init != "kmpp" && init != "random") {
			cout << "Error(KMeans::fit(const MatrixXf&, string, bool)): Invalid init option." << endl;
			exit(1);
		}

		// initialize centers
		if (init == "kmpp") {
			kmpp_init_center(X);
		}
		else{
			rand_init_center(X);
		}

		// indices of objects in each cluster
		vector<vector<int>> prev ;
		while (true) {
			vector<vector<int>> curr = make_clusters(X);

			if (prev == curr)
				break;

			update_centers(X, curr);
			prev = curr;
		}
	}

	void KMeans::kmpp_init_center(const MatrixXf& X)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dist(0, (int)X.rows() - 1);
		
		// get single random center
		centers[0] = X.row(dist(gen));
		for (int i = 1; i < K; i++) {
			vector<float> norms(X.rows());
			for (int j = 0; j < X.rows(); j++) {
				// calculate L2 norm from a point to its closest center
				int idx = nearest_center(X.row(j), i);
				norms[j] = euclidean_norm(X.row(j), centers[idx]);
			}
			int max = (int)std::distance(norms.begin(), std::max_element(norms.begin(), norms.end()));
			centers[i] = X.row(max);
		}
	}

	int KMeans::nearest_center(const RowVectorXf& x, int n_center_initialized)
	{
		vector<float> norms(n_center_initialized);
		for (int i = 0; i < n_center_initialized; i++) {
			norms[i] = euclidean_norm(x, centers[i]);
		}
		return (int)std::distance(norms.begin(), std::min_element(norms.begin(), norms.end()));
	}

	void KMeans::rand_init_center(const MatrixXf& X)
	{
		vector<int> rand_num = generate_random_index((int)X.rows());

		for (int i = 0; i < K; i++) {
			int idx = rand_num[i];
			centers[i] = X.row(idx);
		}
	}

	vector<vector<int>> KMeans::make_clusters(const MatrixXf& X)
	{
		vector<vector<int>> clusters(K);
		for (int i = 0; i < X.rows(); i++) {
			int idx = nearest_center(X.row(i), K);
			clusters[idx].push_back(i);
		}
		return clusters;
	}

	void KMeans::update_centers(const MatrixXf& X, const vector<vector<int>>& clusters)
	{
		for (int i = 0; i < clusters.size(); i++) {
			RowVectorXf sum = RowVectorXf::Zero(X.cols());
			for (int j = 0; j < clusters[i].size(); j++) {
				sum += X.row(clusters[i][j]);
			}
			centers[i] = sum / (float)clusters[i].size();
		}
	}

	vector<vector<int>> KMeans::predict(const MatrixXf& X)
	{
		vector<vector<int>> clusters(K);
		for (int i = 0; i < X.rows(); i++) {
			int idx = nearest_center(X.row(i), K);
			clusters[idx].push_back(i);
		}
		return clusters;
	}

	RowVectorXf* KMeans::get_centers() const { return centers; }
}
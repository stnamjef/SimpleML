#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <numeric>
#include <Eigen/Dense>
#include "common.h"
#include "k_means.h"
using namespace std;
using namespace Eigen;

namespace SimpleML
{
	class GaussianMixture
	{
	private:
		int K;
		MatrixXf posterior;
		RowVectorXf phi;
		RowVectorXf* mu;
		MatrixXf* sigma;
	public:
		GaussianMixture(int K);
		~GaussianMixture();
		void fit(const MatrixXf& X, string init = "kmeans");
		vector<vector<int>> predict(const MatrixXf& X);
	private:
		void random_init(const MatrixXf& X);
		void kmeans_init(const MatrixXf& X);
		void e_step(const MatrixXf& X);
		void m_step(const MatrixXf& X);
	};

	GaussianMixture::GaussianMixture(int K) :
		K(K), mu(nullptr), sigma(nullptr)
	{
		mu = new RowVectorXf[K];
		sigma = new MatrixXf[K];
	}
	
	GaussianMixture::~GaussianMixture()
	{
		delete[] mu;
		delete[] sigma;
	}

	void GaussianMixture::fit(const MatrixXf& X, string init)
	{
		if (init != "kmeans" && init != "random") {
			cout << "Error(GaussianMixture::fit(const MatrixXf, string): Invalid argument." << endl;
			exit(1);
		}

		if (init == "random") 
			random_init(X);
		else 
			kmeans_init(X);

		int i = 1;
		float prev = 0.f, curr;
		while (1) {
			e_step(X);
			m_step(X);

			curr = multivariate_log_likelihood(X, phi, mu, sigma, K);
			
			/*cout << "Epoch " << i << " ]";
			cout << " --> log-likelihood: " << curr << endl;*/

			if ((prev != 0) && (std::fabs(curr - prev) < 0.01))
				break;

			prev = curr;

			if (i > 1000) {
				cout << "Warning(GaussianMixture::fit(const MatrixXf&, int)): ";
				cout << "iteration exceeded 1000 times" << endl;
				break;
			}

			i++;
		}

		cout << endl << "[ Model fit result ]" << endl;
		cout << "Phi: " << endl << phi << endl << endl;
		for (int i = 0; i < K; i++) {
			cout << "Mu" << i + 1 << ": " << endl;
			cout << mu[i] << endl << endl;
			cout << "Sigma" << i + 1 << ": " << endl;
			cout << sigma[i] << endl << endl;
		}
	}

	void GaussianMixture::random_init(const MatrixXf& X)
	{
		int N = (int)X.rows();

		// initialize posterior prob and phi
		posterior = MatrixXf::Constant(N, K, 1.f / K);
		phi = RowVectorXf::Constant(K, 1.f / K);

		// generate random row index
		vector<int> indicies = generate_random_index(N);

		// calculate cov matrix
		MatrixXf cov = covariance_matrix(X);

		// initialize mu and sigma
		for (int i = 0; i < K; i++) {
			mu[i] = X.row(indicies[i]);
			sigma[i] = cov;
		}
	}

	void GaussianMixture::kmeans_init(const MatrixXf& X)
	{
		int N = (int)X.rows();

		// initialize posterior prob
		posterior = MatrixXf::Constant(N, K, 1.f / K);

		KMeans kmeans(K);
		kmeans.fit(X);
		vector<vector<int>> clusters = kmeans.predict(X);

		// initialize phi
		phi.resize(K);
		for (int i = 0; i < K; i++) {
			phi[i] = clusters[i].size() / (float)N;
		}

		const RowVectorXf* centers = kmeans.get_centers();

		// initializ mu and sigma
		for (int i = 0; i < clusters.size(); i++) {
			MatrixXf temp(clusters[i].size(), X.cols());
			for (int j = 0; j < clusters[i].size(); j++) {
				temp.row(j) = X.row(clusters[i][j]);
			}
			mu[i] = centers[i];
			sigma[i] = covariance_matrix(temp);
		}
	}

	void GaussianMixture::e_step(const MatrixXf& X)
	{
		/*
			posterior(i, j) = P(j'th gaussian | x_i)

							  P(j'th gaussian) * P(x_i | j'th gaussian)
							= -----------------------------------------
											P(x_i)

									  phi_j * N(x_i | M_j, S_j)
							= -----------------------------------------
							  Sigma_{k=1}^{K} phi_k * N(x_i | M_k, S_k)
		*/
		for (int i = 0; i < X.rows(); i++) {
			float denominator = 0;
			for (int k = 0; k < K; k++) {
				denominator += phi[k] * multivariate_normal(X.row(i), mu[k], sigma[k]);
			}
			for (int j = 0; j < K; j++) {
				float numerator = phi[j] * multivariate_normal(X.row(i), mu[j], sigma[j]);
				posterior(i, j) = numerator / denominator;
			}
		}
	}

	void GaussianMixture::m_step(const MatrixXf& X)
	{
		int N = (int)X.rows();
		for (int j = 0; j < K; j++) {
			float N_j = posterior.col(j).sum();
			mu[j] = (X.array().colwise() * posterior.col(j).array()).colwise().sum() / N_j;

			MatrixXf centered = X.rowwise() - mu[j];
			sigma[j] = centered.transpose() * posterior.col(j).asDiagonal() * centered / N_j;

			phi[j] = N_j / N;
		}
	}

	vector<vector<int>> GaussianMixture::predict(const MatrixXf& X)
	{
		vector<vector<int>> clusters(K);
		for (int i = 0; i < X.rows(); i++) {
			vector<float> probs(K);
			for (int j = 0; j < K; j++) {
				probs[j] = multivariate_normal(X.row(i), mu[j], sigma[j]);
			}
			int max = (int)std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
			clusters[max].push_back(i);
		}
		return clusters;
	}
}
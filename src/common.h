#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

namespace SimpleML
{
	float euclidean_norm(const RowVectorXf& p1, const RowVectorXf& p2) {
		return std::sqrt((p1 - p2).array().pow(2.f).sum());
	}
	
	vector<int> generate_random_index(int size)
	{
		vector<int> rand_num(size);
		std::iota(rand_num.begin(), rand_num.end(), 0);
		unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(rand_num.begin(), rand_num.end(), std::default_random_engine(seed));
		return rand_num;
	}

	MatrixXf covariance_matrix(const MatrixXf& X)
	{
		MatrixXf centered = X.rowwise() - X.colwise().mean();
		return centered.transpose() * centered / (float)(X.rows() - 1);
	}

	float multivariate_normal(const RowVectorXf& x, const RowVectorXf& mu, const MatrixXf& sigma)
	{
		float inv_sqrt_2pi = 0.3989422804014327f;
		float quad = (x - mu) * sigma.inverse() * (x - mu).transpose();
		float norm = pow(inv_sqrt_2pi, (float)sigma.rows()) * pow(sigma.determinant(), -.5f);
		return norm * exp(-.5f * quad);
	}

	float multivariate_log_likelihood(const MatrixXf& X, const RowVectorXf& phi,
		const RowVectorXf* mu, const MatrixXf* sigma, int K)
	{
		float log_lkhd = 0.f;
		for (int i = 0; i < X.rows(); i++) {
			float lkhd = 0.f;
			for (int j = 0; j < K; j++) {
				lkhd += phi[j] * multivariate_normal(X.row(i), mu[j], sigma[j]);
			}
			log_lkhd += std::log(lkhd);
		}
		return log_lkhd;
	}
}
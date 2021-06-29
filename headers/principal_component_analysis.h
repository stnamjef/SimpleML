#pragma once
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/LU>
using namespace std;
using namespace Eigen;

namespace SimpleML
{
	class PCA
	{
	private:
		int n_component;
		bool is_fitted;
	public:
		MatrixXf U;
		VectorXf S;
	public:
		PCA(int n_component);
		void fit(const MatrixXf& X);
		MatrixXf transform(const MatrixXf& X);
		MatrixXf fit_transform(const MatrixXf& X);
	private:
		void fit_implementation(const MatrixXf& X);
	};
	
	PCA::PCA(int n_component) : n_component(n_component), is_fitted(false) {}

	void PCA::fit(const MatrixXf& X)
	{
		if (X.cols() < n_component) {
			cout << "Error(PCA::fit(const MatrixXf&): The number of features ";
			cout << "must be greater than or equal to the number of components." << endl;
			exit(1);
		}

		fit_implementation(X);
		is_fitted = true;
	}

	void PCA::fit_implementation(const MatrixXf& X)
	{
		/*
			X = U * S * Vt
			X: m x n, U: m x m, S: m x n, Vt: n x n
			SVD solver: 1) Full-SVD, 2) Thin-SVD(default)
		*/
		JacobiSVD<MatrixXf> svd(X.rowwise() - X.colwise().mean(), ComputeThinU | ComputeThinV);
		U = svd.matrixU();
		S = svd.singularValues();
	}

	MatrixXf PCA::transform(const MatrixXf& X)
	{
		if (!is_fitted) {
			cout << "Error(PCA::transform(const MatrixXf&): The model must be fitted first." << endl;
			exit(1);
		}
		if (X.rows() != U.rows()) {
			cout << "Error(PCA::transform(const MatrixXf&): Incompatible feature dimension." << endl;
			exit(1);
		}
		// X_new = X * V = U * S * Vt * V = U * S
		return U.topLeftCorner(U.rows(), n_component) * S.head(n_component).asDiagonal();
	}

	MatrixXf PCA::fit_transform(const MatrixXf& X)
	{
		fit_implementation(X);
		return U.topLeftCorner(U.rows(), n_component) * S.head(n_component).asDiagonal();
	}
}
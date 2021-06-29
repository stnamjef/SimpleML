#pragma once
#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

namespace SimpleML
{
	class OLS
	{
	private:
		VectorXf coeffs;
	public:
		OLS();
		void fit(const MatrixXf& A, const VectorXf& b);
		VectorXf predict(const MatrixXf& A);
	};

	OLS::OLS() {};

	void OLS::fit(const MatrixXf& A, const VectorXf& b)
	{
		/*
			---------- OLS ----------
			Ax = b
			(At * A)x = At * b
			x = (At * A)^-1 * At * b
			------ OLS with SVD -----
			A = U * S * Vt
			A+ = V * S+ * Ut (pseudo inverse)
			x = A+ * b
			---------- Note ---------
			(At * A)^-1 * At = V * (St * S)^-1 * St * Vt
		*/

		JacobiSVD<MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
		coeffs = svd.solve(b);

		cout << "OLS coefficients: ";
		cout << coeffs.transpose() << endl;
	}

	VectorXf OLS::predict(const MatrixXf& A)
	{
		if (A.cols() != coeffs.size()) {
			cout << "LinearRegression::predict(const MatrixXf&): Invalid matrix size." << endl;
			exit(1);
		}
		return A * coeffs;
	}
}
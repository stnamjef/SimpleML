#include <iostream>
#include <iomanip>
#include <string>
#include <Eigen/Dense>
#include "./headers/file_manage.h"
#include "./headers/model_evaluation.h"
#include "./headers/k_nearest_neighbors.h"
#include "./headers/naive_bayes.h"
#include "./headers/decision_tree.h"
#include "./headers/principal_component_analysis.h"
#include "./headers/k_means.h"
#include "./headers/gaussian_mixture.h"
#include "./headers/ordinary_least_squares.h"
using namespace std;
using namespace Eigen;

// decision tree classifier
int main()
{
	string file_name = "./dataset/iris.csv";

	MatrixXf X;
	VectorXi Y;

	// read csv
	SimpleML::read_csv(file_name, X, Y);

	// run decision tree classifier
	SimpleML::DecisionTree dt;
	dt.fit(X, Y);
	dt.print_tree();

	// evaluate
	VectorXi predicted = dt.predict(X);
	float acc = SimpleML::calc_accuracy(Y, predicted);
	cout << "\nAccuracy: " << acc * 100 << "%" << endl;

	// K-fold cross validation. Uncomment below if necessary
	/*float acc = SimpleML::evaluate_classification_model(dt, X, Y, 5);
	cout << "Accuracy(k-fold): " << acc * 100 << "%" << endl;*/
}

//// naive bayes
//int main()
//{
//	string file_name = "./dataset/iris.csv";
//
//	MatrixXf X;
//	VectorXi Y;
//
//	// read csv
//	SimpleML::read_csv(file_name, X, Y);
//
//	// run naive bayes classifier
//	SimpleML::NaiveBayes nb;
//	nb.fit(X, Y);
//
//	// evaluate
//	VectorXi predicted = nb.predict(X);
//	float acc = SimpleML::calc_accuracy(Y, predicted);
//	cout << "Accuracy: " << acc * 100 << "%" << endl;
//
//	// K-fold cross validation. Uncomment below if necessary
//	//float acc = SimpleML::evaluate_classification_model(nb, X, Y, 5);
//	//cout << "Accuracy(k-fold): " << acc * 100 << "%" << endl;
//
//	return 0;
//}

//// k-nearest neighbors
//int main()
//{
//	string file_name = "./dataset/iris.csv";
//
//	MatrixXf X;
//	VectorXi Y;
//
//	// read csv
//	SimpleML::read_csv(file_name, X, Y);
//
//	// run k-nearest neighbors
//	SimpleML::KNN knn(4);
//	knn.fit(X, Y);
//
//	// evaluate
//	VectorXi predicted = knn.predict(X);
//	float acc = SimpleML::calc_accuracy(Y, predicted);
//	cout << "Accuracy: " << acc * 100 << "%" << endl;
//
//	// K-fold cross validation. Uncomment below if necessary
//	//float acc = SimpleML::evaluate_classification_model(knn, X, Y, 5);
//	//cout << "Accuracy(k-fold): " << acc * 100 << "%" << endl;
//
//	return 0;
//}

//// OLS
//int main()
//{
//	string file_name = "./dataset/winequality-white.csv";
//
//	MatrixXf X;
//	VectorXf Y;
//	
//	// read csv
//	SimpleML::read_csv(file_name, X, Y);
//
//	// add constant term if necessary.
//	bool add_constant = true;
//	if (add_constant) {
//		X = SimpleML::add_constant(X);
//	}
//
//	// run OLS.
//	SimpleML::OLS ols;
//	ols.fit(X, Y);
//	
//	// predict
//	VectorXf predicted = ols.predict(X);
//	cout << "Predicted:" << endl;
//	cout << predicted << endl;
//
//	return 0;
//}

//// gaussian mixture
//int main()
//{
//	string file_name = "./dataset/iris.csv";
//
//	MatrixXf X;
//	VectorXi Y;
//
//	// read csv
//	SimpleML::read_csv(file_name, X, Y);
//
//	// run gaussian mixture model
//	SimpleML::GaussianMixture gm(3);
//	gm.fit(X);
//
//	// evaluate
//	vector<vector<int>> predicted = gm.predict(X);
//	float sil = SimpleML::silhouette_score(X, predicted);
//	cout << "Silhouette score : " << sil << endl;
//
//	return 0;
//}

//// K-means clustering
//int main()
//{
//	string file_name = "./dataset/iris.csv";
//
//	MatrixXf X;
//	VectorXi Y;
//
//	// read csv
//	SimpleML::read_csv(file_name, X, Y);
//
//	// run k-means
//	SimpleML::KMeans km(3);
//	km.fit(X);
//
//	// evaluate
//	vector<vector<int>> predicted = km.predict(X);
//	float sil = SimpleML::silhouette_score(X, predicted);
//	cout << "Silhouette score : " << sil << endl;
//
//	return 0;
//}

//// pricipal component analysis
//int main()
//{
//	string file_name = "./dataset/iris.csv";
//
//	MatrixXf X;
//	VectorXi Y;
//
//	// read csv
//	SimpleML::read_csv(file_name, X, Y);
//
//	// run pca
//	SimpleML::PCA pca(2);
//	MatrixXf Xt = pca.fit_transform(X);
//
//	cout << "Dimensionality reduction result:" << endl;
//	cout << Xt << endl;
//
//	return 0;
//}
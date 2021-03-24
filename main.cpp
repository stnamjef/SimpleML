#include <iostream>
#include <iomanip>
#include <string>
#include <Eigen/Dense>
#include "file_manage.h"
#include "model_evaluation.h"
#include "k_nearest_neighbors.h"
#include "naive_bayes.h"
#include "decision_tree.h"
#include "principal_component_analysis.h"
#include "k_means.h"
#include "gaussian_mixture.h"
using namespace std;
using namespace Eigen;


int main()
{
	string file_name = "./data/iris.csv";

	MatrixXf X;
	VectorXi Y;
	
	int n_row = 150, n_col = 5;

	// read csv
	SimpleML::read_csv(file_name, X, Y, n_row, n_col);

	/*SimpleML::KNN knn(4);
	auto start = std::chrono::system_clock::now();
	float accuracy = SimpleML::evaluate_classification_model(knn, X, Y, 5);
	auto end = std::chrono::system_clock::now();
	auto mill = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	cout << "KNN Classifier - accuracy: " << setprecision(3) << accuracy;
	cout << ", t: " << mill.count() << "ms" << endl;
	
	SimpleML::NaiveBayes naive_bayes;
	start = std::chrono::system_clock::now();
	accuracy = SimpleML::evaluate_classification_model(naive_bayes, X, Y, 5);
	end = std::chrono::system_clock::now();
	mill = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	cout << "Naive Bayes Classifier - accuracy: " << setprecision(3) << accuracy;
	cout << ", t: " << mill.count() << "ms" << endl;


	SimpleML::DecisionTree decision_tree;
	start = std::chrono::system_clock::now();
	accuracy = SimpleML::evaluate_classification_model(decision_tree, X, Y, 5);
	end = std::chrono::system_clock::now();
	mill = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	cout << "Decision Tree Classifier - accuracy: " << setprecision(3) << accuracy;
	cout << ", t: " << mill.count() << "ms" << endl;

	SimpleML::PCA pca(2);
	start = std::chrono::system_clock::now();
	MatrixXf transformed = pca.fit_transform(X);
	end = std::chrono::system_clock::now();
	mill = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	cout << "Principal Component Analysis - t: " << mill.count() << "ms" << endl;

	SimpleML::KMeans kmeans(3);
	start = std::chrono::system_clock::now();
	float silhouette = SimpleML::evaluate_clustering_model(kmeans, X);
	end = std::chrono::system_clock::now();
	mill = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	cout << "K-Means clustering - silhouette score: " << setprecision(3) << silhouette;
	cout << ", t: " << mill.count() << "ms" << endl;*/

	SimpleML::GaussianMixture gm(3);
	float silhouette = SimpleML::evaluate_clustering_model(gm, X);
	cout << silhouette << endl;


	return 0;
}
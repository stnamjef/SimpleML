# SimpleML

- SimpleML is a C++ implementation of various machine learning algorithms.
- The code is simple and easy to understand.
- It can help you get a clear understanding of machine learning algorithms.
- [A dockerized test environment](https://hub.docker.com/repository/docker/stnamjef/eigen_3.3.9) is available.

## 1. Requirements

- Eigen 3.3.9
- g++ 9.0 or higher

## 2. Supported ML algorithms

- Classification
  - Decision tree
  - Naive Bayes
  - K-nearest neighbors

- Clustering
  - K-means
  - Gaussian mixture
- Dimensionality reduction
  - Principal Component Analysis (PCA)
- Regression
  - Ordinary Least Squares (OLS)

## 3. Examples

- Example code of all models is already written in main.cpp file.
- Please uncomment the code you want to use and check if file_name (path to dataset) is correct.
- Ex 1) Decision tree classifier

```c++
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "./headers/file_manage.h"
#include "./headers/model_evaluation.h"
#include "./headers/decision_tree.h"
using namespace std;
using namespace Eigen;

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
```

- Ex 2) Gaussian mixture

```c++
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "./headers/file_manage.h"
#include "./headers/model_evaluation.h"
#include "./headers/gaussian_mixture.h"
using namespace std;
using namespace Eigen;

int main()
{
	string file_name = "./dataset/iris.csv";

	MatrixXf X;
	VectorXi Y;

	// read csv
	SimpleML::read_csv(file_name, X, Y);

	// run gaussian mixture model
	SimpleML::GaussianMixture gm(3);
	gm.fit(X);

	// evaluate
	vector<vector<int>> predicted = gm.predict(X);
	float sil = SimpleML::silhouette_score(X, predicted);
	cout << "Silhouette score : " << sil << endl;

	return 0;
}
```

## 4. Compile & Run

- Pull the docker image

```shell
# host shell
docker pull stnamjef/eigen_3.3.9:1.0
```

- Run the docker image

```shell
# pwd -> the project directory (SimpleML)
docker run -it -v $(pwd):/usr/build stnamjef/eigen_3.3.9:1.0
```

- Compile

```shell
# container shell at /usr/build
g++ main.cpp --std=c++17 -O2 -o simpleml
```

- Run

```shell
./simpleml
```

## 5. Ongoing list

- CLI options
- Logistic regression, SVM
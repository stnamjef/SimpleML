#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

namespace SimpleML {
	void read_csv(string file_name, MatrixXf& features, VectorXi& labels, int n_row, int n_col)
		// Warning: The last column must be a class vector
	{
		ifstream fin(file_name);
		if (!fin) {
			cout << "Error(read_csv(string, MatrixXf&, VectorXf&, int, int): File not found." << endl;
			exit(1);
		}
		else {
			// resize matricies
			features.resize(n_row, n_col - 1);
			labels.resize(n_row);

			string line;
			int row = 0, col = 0;
			vector<string> str_labels;
			while (std::getline(fin, line)) {
				string str;
				stringstream ss(line);
				while (std::getline(ss, str, ',')) {
					if (col >= n_col) {
						cout << "Error(read_csv(string, MatrixXf&, VectorXf&, int, int): Invalid column size." << endl;
						exit(1);
					}
					if (col + 1 == n_col) {
						auto iter = std::find(str_labels.begin(), str_labels.end(), str);
						if (iter != str_labels.end()) {
							labels[row] = (int)std::distance(str_labels.begin(), iter);
						}
						else {
							str_labels.push_back(str);
							labels[row] = (int)str_labels.size() - 1;
						}
						row++;
						col = 0;
					}
					else {
						features(row, col) = std::stof(str);
						col++;
					}
				}
				if (row > n_row) {
					cout << "Error(read_csv(string, MatrixXf&, VectorXf&, int, int): Invalid row size." << endl;
					exit(1);
				}
			}
		}
	}
}
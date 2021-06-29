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
	void count_row_col(string file_name, int& n_row, int& n_col)
	{
		ifstream fin(file_name);
		if (!fin) {
			cout << "Error(read_csv(string, MatrixXf&, VectorXf&, int, int): File not found." << endl;
			exit(1);
		}

		n_row = 0;
		n_col = 0;

		int col = 0;
		string line;
		while (std::getline(fin, line)) {
			string str;
			stringstream ss(line);
			while (std::getline(ss, str, ',')) {
				col++;
			}
			n_row++;

			if (n_col != 0 && n_col != col) {
				cout << "Error(count_row_col(string, int&, int&): All rows should have the same length." << endl;
				fin.close();
				exit(1);
			}

			n_col = col;
			col = 0;
		}
		fin.close();
	}

	bool is_number(const string& str)
	{
		for (const char& c : str) {
			if (!std::isdigit(c) && c != '.') {
				return false;
			}
		}
		return true;
	}

	template<class T>
	void read_csv(string file_name, MatrixXf& features, Matrix<T, Dynamic, 1>& labels)
	{
		int n_row, n_col;
		count_row_col(file_name, n_row, n_col);
		
		// because the first row is column name, and the last column is label
		n_row--;
		n_col--;

		// resize matricies
		features.resize(n_row, n_col);
		labels.resize(n_row);

		// open file
		ifstream fin(file_name);

		// read the first line (column names)
		string line;
		getline(fin, line);

		int row = 0, col = 0;
		vector<string> str_labels;
		while (fin >> line) {
			string str;
			stringstream ss(line);

			// read features
			while (std::getline(ss, str, ',')) {
				if (col == n_col - 1) {
					col = 0;
					break;
				}
				features(row, col) = std::stof(str);
				col++;
			}

			// read labels
			if (is_number(str)) {
				labels[row] = std::stof(str);
			}
			else {
				auto iter = std::find(str_labels.begin(), str_labels.end(), str);
				if (iter != str_labels.end()) {
					labels[row] = (int)std::distance(str_labels.begin(), iter);
				}
				else {
					str_labels.push_back(str);
					labels[row] = (int)str_labels.size() - 1;
				}
			}

			row++;
		}

		fin.close();
	}
}
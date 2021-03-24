#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

namespace SimpleML
{
	struct Question
	{
		int col;
		float value;
		float gain;
		Question() : col(0), value(0), gain(0) {}
		Question(int col, float value) : col(col), value(value), gain(0) {}
		bool match(const RowVectorXf& x) const { return x[col] >= value; }
		Question& operator=(const Question& other) 
		{
			col = other.col;
			value = other.value;
			gain = other.gain;
			return *this;
		}
	};

	struct Node
	{
		Question Q;
		Node* left = nullptr;
		Node* right = nullptr;
		vector<int> labels;
		~Node()
		{
			delete left;
			delete right;
		}
	};

	class DecisionTree
	{
	private:
		Node* root;
		int n_class;
	public:
		DecisionTree();
		~DecisionTree();
		void fit(const MatrixXf& X, const VectorXi& Y);
		VectorXi predict(const MatrixXf& X);
		void print_tree();
	};

	void build_tree(Node*& node, const MatrixXf& X, const VectorXi& Y,
		const vector<int>& split, const vector<int>& cols);

	Question find_best_question(const MatrixXf& X, const VectorXi& Y,
		const vector<int>& split, const vector<int>& cols);

	float gini(const VectorXi& Y, const vector<int>& split);

	float entropy(const VectorXi& Y, const vector<int>& split);

	vector<int> count_class(const VectorXi& Y, const vector<int>& split);

	vector<float> get_unique_values(const VectorXf& col, const vector<int>& split);

	vector<vector<int>> split_node(const Question& Q, const MatrixXf& X, const vector<int>& split);

	float info_gain(const VectorXi& Y, const vector<int>& left, const vector<int>& right, float current_impurity);
	
	vector<int> erase_taken_col(const Question& Q, const vector<int>& cols);

	Node* find_leaf_node(const RowVectorXf& x, Node* node);

	void print_implementation(Node* node, int width);

	/*---------------------------------------------------------------------------------------*/

	DecisionTree::DecisionTree() : root(nullptr), n_class(0) {}

	DecisionTree::~DecisionTree() { delete root; }

	void DecisionTree::fit(const MatrixXf& X, const VectorXi& Y)
	{
		vector<int> init_split(X.rows());
		std::iota(init_split.begin(), init_split.end(), 0);

		vector<int> init_cols(X.cols());
		std::iota(init_cols.begin(), init_cols.end(), 0);

		build_tree(root, X, Y, init_split, init_cols);
	}

	void build_tree(Node*& node, const MatrixXf& X, const VectorXi& Y,
		const vector<int>& split, const vector<int>& cols)
	{
		node = new Node;
		node->labels = count_class(Y, split);

		Question Q = find_best_question(X, Y, split, cols);

		if (Q.gain >= 0.2) {
			node->Q = Q;

			vector<vector<int>> splits = split_node(Q, X, split);
			vector<int> new_cols = erase_taken_col(Q, cols);

			build_tree(node->left, X, Y, splits[0], new_cols);
			build_tree(node->right, X, Y, splits[1], new_cols);
		}
	}

	Question find_best_question(const MatrixXf& X, const VectorXi& Y,
		const vector<int>& split, const vector<int>& cols)
	{
		Question best_Q;
		best_Q.gain = 0;
		
		float current_impurity = gini(Y, split);
		for (int idx : cols) {
			vector<float> unique = get_unique_values(X.col(idx), split);
			for (float value : unique) {
				Question Q(idx, value);
				vector<vector<int>> splits = split_node(Q, X, split);
				if (splits[0].size() == 0 || splits[1].size() == 0) {
					continue;
				}
				Q.gain = info_gain(Y, splits[0], splits[1], current_impurity);
				if (Q.gain >= best_Q.gain) {
					best_Q = Q;
				}
			}
		}
		return best_Q;
	}

	float gini(const VectorXi& Y, const vector<int>& split)
	{
		vector<int> counts = count_class(Y, split);

		// calculate gini
		float impurity = 1;
		float split_size = (float)split.size();
		for (int count : counts) {
			impurity -= std::pow(count / split_size, 2.f);
		}
		return impurity;
	}

	float entropy(const VectorXi& Y, const vector<int>& split)
	{
		vector<int> counts = count_class(Y, split);

		// calculate entropy
		float impurity = 0;
		float split_size = (float)split.size();
		for (int count : counts) {
			float prob = count / split_size;
			impurity -= prob * std::log2(prob);
		}
		return impurity;
	}

	vector<int> count_class(const VectorXi& Y, const vector<int>& split)
	{
		// count class
		int n_class = (int)(*std::max_element(Y.data(), Y.data() + Y.size()) + 1);
		vector<int> counts(n_class);
		for (int idx : split) {
			int label = (int)Y[idx];
			counts[label]++;
		}
		return counts;
	}

	vector<float> get_unique_values(const VectorXf& col, const vector<int>& split)
	{
		vector<float> unique;
		for (int idx : split) {
			// check if overlap
			auto iter = std::find(unique.begin(), unique.end(), col[idx]);
			if (iter == unique.end()) {
				unique.push_back(col[idx]);
			}
		}
		return unique;
	}

	vector<vector<int>> split_node(const Question& Q, const MatrixXf& X, const vector<int>& split)
	{
		vector<vector<int>> splits(2);
		for (int idx : split) {
			if (Q.match(X.row(idx)))
				splits[0].push_back(idx);
			else
				splits[1].push_back(idx);
		}
		return splits;
	}

	float info_gain(const VectorXi& Y, const vector<int>& left, const vector<int>& right, float current_impurity)
	{
		float P = (float)left.size() / (left.size() + right.size());
		return current_impurity - P * gini(Y, left) - (1 - P) * gini(Y, right);
	}

	vector<int> erase_taken_col(const Question& Q, const vector<int>& cols)
	{
		vector<int> new_cols;
		for (int idx : cols) {
			if (idx != Q.col)
				new_cols.push_back(idx);
		}
		return new_cols;
	}

	VectorXi DecisionTree::predict(const MatrixXf& X)
	{
		VectorXi labels(X.rows());
		for (int i = 0; i < X.rows(); i++) {
			Node* leaf = find_leaf_node(X.row(i), root);
			auto max = std::max_element(leaf->labels.begin(), leaf->labels.end());
			labels[i] = (int)std::distance(leaf->labels.begin(), max);
		}
		return labels;
	}

	Node* find_leaf_node(const RowVectorXf& x, Node* node)
	{
		while (node->left != nullptr && node->right != nullptr) {
			if (node->Q.match(x)) {
				node = node->left;
			}
			else {
				node = node->right;
			}
		}
		return node;
	}

	void DecisionTree::print_tree() { print_implementation(root, 0); }

	void print_implementation(Node* node, int width)
	{
		if (node->left == nullptr && node->right == nullptr) {
			cout << setw(width + 4) << " " << "Predict : {";
			for (int i = 0; i < node->labels.size(); i++)
				cout << "'" << i << "' : " << node->labels[i] << ", ";
			cout << "}" << endl;
		}
		else {
			cout << setw(width) << " " << "Q : X" << node->Q.col + 1 << " >= " <<
				node->Q.value << " ? " << endl;

			cout << setw(width) << " " << "--> True: " << endl;
			print_implementation(node->left, width + 4);

			cout << setw(width) << " " << "--> False: " << endl;
			print_implementation(node->right, width + 4);
		}
	}
}
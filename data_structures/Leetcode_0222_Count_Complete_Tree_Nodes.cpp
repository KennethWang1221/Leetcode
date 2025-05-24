#include <iostream>
#include <vector>

using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
private:
    // Helper to get height of leftmost path
    int leftHeight(TreeNode* node) {
        int height = 0;
        while (node) {
            height++;
            node = node->left;
        }
        return height;
    }

    // Helper to get height of rightmost path
    int rightHeight(TreeNode* node) {
        int height = 0;
        while (node) {
            height++;
            node = node->right;
        }
        return height;
    }

public:
    int countNodes(TreeNode* root) {
        if (!root) return 0;

        int left = leftHeight(root);
        int right = rightHeight(root);

        if (left == right) {
            return (1 << left) - 1; // 2^h - 1
        } else {
            return 1 + countNodes(root->left) + countNodes(root->right);
        }
    }
};

// Helper to build sample tree
TreeNode* buildSampleTree() {
    TreeNode* n1 = new TreeNode(1);
    TreeNode* n2 = new TreeNode(2);
    TreeNode* n3 = new TreeNode(3);
    TreeNode* n4 = new TreeNode(4);
    TreeNode* n5 = new TreeNode(5);
    TreeNode* n6 = new TreeNode(6);

    n1->left = n2;
    n1->right = n3;
    n2->left = n4;
    n2->right = n5;
    n3->left = n6;

    return n1;
}

int main() {
    Solution sol;
    TreeNode* root = buildSampleTree();

    cout << "Total Nodes: " << sol.countNodes(root) << endl; // Output: 6

    return 0;
}

// g++ -std=c++17 Leetcode_0222_Count_Complete_Tree_Nodes.cpp -o test
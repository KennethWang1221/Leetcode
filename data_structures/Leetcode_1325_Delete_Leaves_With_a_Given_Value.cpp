#include <iostream>
using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    TreeNode* removeLeafNodes(TreeNode* root, int target) {
        // Base case: if the node is NULL, return NULL
        if (!root) {
            // If the current node is nullptr, return nullptr
            return nullptr;
        }

        // Recursively process the left subtree
        root->left = removeLeafNodes(root->left, target);
        // Recursively process the right subtree
        root->right = removeLeafNodes(root->right, target);

        // Check if the current node is a leaf node with the target value
        if (!root->left && !root->right && root->val == target) {
            // If true, return nullptr to remove this node
            return nullptr;
        }

        // If the current node is not a target leaf node, return the current
        // node
        return root;
    }
};

// Helper function to print the tree in-order (for validation)
void printTree(TreeNode* root) {
    if (root != nullptr) {
        printTree(root->left);
        cout << root->val << " ";
        printTree(root->right);
    }
}

int main() {
    // Creating a sample binary tree
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(2);
    root->left->right = new TreeNode(4);
    root->right->right = new TreeNode(2);

    Solution sol;

    // Test case 1: Removing leaf nodes with value 2
    cout << "Original tree (in-order): ";
    printTree(root);
    cout << endl;

    root = sol.removeLeafNodes(root, 2);

    cout << "After removing leaf nodes with value 2 (in-order): ";
    printTree(root);
    cout << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_1325_Delete_Leaves_With_a_Given_Value.cpp -o test
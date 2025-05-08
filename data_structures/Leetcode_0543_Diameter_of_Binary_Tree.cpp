#include <iostream>
#include <algorithm>
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
    int diameter = 0; // To store the maximum diameter

    int depth(TreeNode* node) {
        if (!node) return 0; // Base case: if the node is null, depth is 0

        int left_depth = depth(node->left);   // Get the depth of the left subtree
        int right_depth = depth(node->right); // Get the depth of the right subtree

        // Update the diameter (the longest path through this node)
        diameter = max(diameter, left_depth + right_depth);

        return max(left_depth, right_depth) + 1; // Return the height of the subtree
    }

    int diameterOfBinaryTree(TreeNode* root) {
        depth(root);  // Perform DFS from the root
        return diameter;  // Return the maximum diameter found
    }
};

// Helper function to print the result (For testing purposes)
int main() {
    // Constructing the tree:
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);

    Solution sol;
    cout << "Diameter of the binary tree: " << sol.diameterOfBinaryTree(root) << endl;  // Expected Output: 3

    return 0;
}


// g++ -std=c++17 Leetcode_0543_Diameter_of_Binary_Tree.cpp -o test

#include <iostream>
#include <limits.h>
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
    bool isValidBST(TreeNode* root) {
        return valid(root, LONG_MIN, LONG_MAX);
    }
    
private:
    bool valid(TreeNode* node, long left, long right) {
        if (!node) {
            return true; // A null node is valid
        }

        // Check if the current node's value is within the valid range
        if (node->val <= left || node->val >= right) {
            return false; // The value is out of valid range
        }

        // Recursively check the left and right subtrees
        return valid(node->left, left, node->val) && valid(node->right, node->val, right);
    }
};

// Helper function to print the result
int main() {
    // Test case: Construct the binary tree
    TreeNode* n1 = new TreeNode(2);
    TreeNode* n2 = new TreeNode(1);
    TreeNode* n3 = new TreeNode(3);
    
    n1->left = n2;
    n1->right = n3;

    // Create an instance of the Solution class and test
    Solution sol;
    bool res = sol.isValidBST(n1);

    cout << (res ? "True" : "False") << endl;  // Expected output: True

    return 0;
}


// g++ -std=c++17 Leetcode_0098_Validate_Binary_Search_Tree.cpp -o test

#include <iostream>
#include <algorithm>
#include <climits>

using namespace std;

class TreeNode {
public:
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    int maxPathSum(TreeNode* root) {
        int res = root->val;

        // Start DFS traversal from the root node
        helper(root, res);

        return res;
    }

private:
    // Helper function for DFS traversal
    int helper(TreeNode* node, int &res) {
        if (!node) {
            return 0;
        }

        // Max path sum on the left and right, with 0 if negative
        int leftMax = max(helper(node->left, res), 0);
        int rightMax = max(helper(node->right, res), 0);

        // Update result with the sum including the current node
        res = max(res, node->val + leftMax + rightMax);

        // Return the maximum path sum including the current node
        return node->val + max(leftMax, rightMax);
    }
};

int main() {
    TreeNode* n1 = new TreeNode(-10);
    TreeNode* n2 = new TreeNode(9);
    TreeNode* n3 = new TreeNode(20);
    TreeNode* n4 = new TreeNode(15);
    TreeNode* n5 = new TreeNode(7);

    n1->left = n2;
    n1->right = n3;

    n3->left = n4;
    n3->right = n5;

    Solution solution;
    cout << solution.maxPathSum(n1) << endl;

    // Clean up the allocated memory
    delete n1;
    delete n2;
    delete n3;
    delete n4;
    delete n5;

    return 0;
}



// g++ -std=c++17 Leetcode_0124_Binary_Tree_Maximum_Path_Sum.cpp -o test
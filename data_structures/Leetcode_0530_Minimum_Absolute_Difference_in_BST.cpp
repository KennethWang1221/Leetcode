#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>

using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    int getMinimumDifference(TreeNode* root) {
        vector<int> res;
        if (!root)
            return 0;

        queue<TreeNode*> q;
        q.push(root);

        // Step 1: BFS to collect all node values
        while (!q.empty()) {
            int levelSize = q.size();
            for (int i = 0; i < levelSize; ++i) {
                TreeNode* cur = q.front(); q.pop();
                res.push_back(cur->val);
                if (cur->left)
                    q.push(cur->left);
                if (cur->right)
                    q.push(cur->right);
            }
        }

        // Step 2: Sort and compute min diff
        sort(res.begin(), res.end());
        int min_diff = INT_MAX;
        for (int i = 1; i < res.size(); ++i) {
            min_diff = min(min_diff, abs(res[i] - res[i - 1]));
        }

        return min_diff;
    }
};

int main() {
    // Build the sample tree:
    //         4
    //        / \
    //       2   6
    //      / \
    //     1   3

    TreeNode* root = new TreeNode(4);
    root->left = new TreeNode(2);
    root->right = new TreeNode(6);
    root->left->left = new TreeNode(1);
    root->left->right = new TreeNode(3);

    Solution sol;
    int result = sol.getMinimumDifference(root);
    cout << "Minimum Difference: " << result << endl; // Expected: 1
    return 0;
}

// g++ -std=c++17 Leetcode_0530_Minimum_Absolute_Difference_in_BST.cpp -o test
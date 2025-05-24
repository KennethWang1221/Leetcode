#include <iostream>
#include <vector>
#include <stack>
#include <string>
#include <sstream>

using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    // Constructors
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* l, TreeNode* r) : val(x), left(l), right(r) {}
};

class Solution {
public:
    int sumNumbers(TreeNode* root) {
        if (!root) return 0;

        stack<TreeNode*> nodeStack;
        stack<string> pathStack;
        vector<int> res;

        nodeStack.push(root);
        pathStack.push(to_string(root->val));

        while (!nodeStack.empty()) {
            TreeNode* cur = nodeStack.top(); nodeStack.pop();
            string path = pathStack.top(); pathStack.pop();

            if (!cur->left && !cur->right) {
                res.push_back(stoi(path));
                continue;
            }

            if (cur->right) {
                nodeStack.push(cur->right);
                pathStack.push(path + to_string(cur->right->val));
            }

            if (cur->left) {
                nodeStack.push(cur->left);
                pathStack.push(path + to_string(cur->left->val));
            }
        }

        int total = 0;
        for (int num : res)
            total += num;

        return total;
    }
};

// Test Case
int main() {
    // Build the tree:
    //         4
    //        / \
    //       9   0
    //      / \
    //     5   1

    TreeNode n5(5); // leaf
    TreeNode n1(1); // leaf
    TreeNode n9(9, &n5, &n1);
    TreeNode n0(0);
    TreeNode root(4, &n9, &n0);

    Solution sol;
    cout << "Sum of root-to-leaf numbers: " << sol.sumNumbers(&root) << endl; // Expected: 495 + 491 + 40 = 1026

    return 0;
}
// g++ -std=c++17 Leetcode_0129_Sum_Root_to_Leaf_Numbers.cpp -o test
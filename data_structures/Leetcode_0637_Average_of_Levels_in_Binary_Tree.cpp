#include <iostream>
#include <vector>
#include <queue>

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
    vector<double> averageOfLevels(TreeNode* root) {
        vector<double> result;
        if (!root) return result;

        queue<TreeNode*> q;
        q.push(root);

        while (!q.empty()) {
            int levelSize = q.size();
            double levelSum = 0;

            for (int i = 0; i < levelSize; ++i) {
                TreeNode* cur = q.front(); q.pop();
                levelSum += cur->val;

                if (cur->left)
                    q.push(cur->left);
                if (cur->right)
                    q.push(cur->right);
            }

            result.push_back(levelSum / levelSize);
        }

        return result;
    }
};

int main() {
    // Build sample binary tree
    TreeNode* root = new TreeNode(3);
    root->left = new TreeNode(9);
    root->right = new TreeNode(20);
    root->right->left = new TreeNode(15);
    root->right->right = new TreeNode(7);

    Solution sol;
    vector<double> res = sol.averageOfLevels(root);

    cout << "[ ";
    for (double avg : res) {
        cout << avg << " ";
    }
    cout << "]" << endl;

    return 0;
}
// g++ -std=c++17 Leetcode_0637_Average_of_Levels_in_Binary_Tree.cpp -o test
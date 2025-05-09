#include <iostream>
#include <queue>
#include <vector>
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
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> results;
        if (!root) {
            return results;
        }

        queue<TreeNode*> que;
        que.push(root);

        while (!que.empty()) {
            int size = que.size();
            vector<int> result;

            for (int i = 0; i < size; ++i) {
                TreeNode* cur = que.front();
                que.pop();
                result.push_back(cur->val);

                if (cur->left) {
                    que.push(cur->left);
                }

                if (cur->right) {
                    que.push(cur->right);
                }
            }

            results.push_back(result);
        }

        return results;
    }
};

int main() {
    TreeNode* n1 = new TreeNode(3);
    TreeNode* n2 = new TreeNode(9);
    TreeNode* n3 = new TreeNode(20);
    TreeNode* n4 = new TreeNode(-1);  // Use -1 for non-existing nodes if needed
    TreeNode* n5 = new TreeNode(-1);
    TreeNode* n6 = new TreeNode(15);
    TreeNode* n7 = new TreeNode(7);

    n1->left = n2;
    n1->right = n3;
    n2->left = n4;
    n2->right = n5;
    n3->left = n6;
    n3->right = n7;

    Solution s;
    vector<vector<int>> res = s.levelOrder(n1);

    for (const auto& level : res) {
        for (int val : level) {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}



// g++ -std=c++17 Leetcode_0102_Binary_Tree_Level_Order_Traversal.cpp -o test

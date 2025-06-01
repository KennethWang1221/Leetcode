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
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if (!root) return res;

        queue<TreeNode*> q;
        q.push(root);
        int level = 0;

        while (!q.empty()) {
            int level_size = q.size();
            vector<int> curr_level(level_size);

            for (int i = 0; i < level_size; ++i) {
                TreeNode* cur = q.front(); q.pop();

                // In zigzag, we fill left-to-right or right-to-left
                int index = (level % 2 == 0) ? i : (level_size - 1 - i);
                curr_level[index] = cur->val;

                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
            }

            res.push_back(curr_level);
            level++;
        }

        return res;
    }
};

// Helper to build sample tree
TreeNode* buildSampleTree() {
    TreeNode* root = new TreeNode(3);
    root->left = new TreeNode(9);
    root->right = new TreeNode(20);
    root->right->left = new TreeNode(15);
    root->right->right = new TreeNode(7);
    return root;
}

// Print result helper
void printResult(const vector<vector<int>>& res) {
    cout << "[\n";
    for (const auto& level : res) {
        cout << "  [";
        for (int i = 0; i < level.size(); ++i) {
            cout << level[i];
            if (i != level.size() - 1)
                cout << ", ";
        }
        cout << "]\n";
    }
    cout << "]" << endl;
}

int main() {
    TreeNode* root = buildSampleTree();
    Solution sol;
    vector<vector<int>> res = sol.zigzagLevelOrder(root);

    printResult(res); // Expected: [[3], [20, 9], [15, 7]]

    return 0;
}

// g++ -std=c++17 Leetcode_0103_Binary_Tree_Zigzag_Level_Order_Traversal.cpp -o test
#include <iostream>
#include <queue>
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
    int maxDepth(TreeNode* root) {
        if (!root) return 0;

        queue<TreeNode*> que;
        que.push(root);
        int depth = 0;

        while (!que.empty()) {
            int n = que.size();
            for (int i = 0; i < n; ++i) {
                TreeNode* cur = que.front();
                que.pop();

                if (cur->left) que.push(cur->left);
                if (cur->right) que.push(cur->right);
            }
            depth++;
        }
        return depth;
    }
};

// Helper function to create a binary tree
TreeNode* createTree() {
    TreeNode* n1 = new TreeNode(3);
    TreeNode* n2 = new TreeNode(9);
    TreeNode* n3 = new TreeNode(20);
    TreeNode* n4 = new TreeNode(15);
    TreeNode* n5 = new TreeNode(7);

    n1->left = n2;
    n1->right = n3;
    n3->left = n4;
    n3->right = n5;

    return n1;
}

int main() {
    Solution solution;

    // Create the tree
    TreeNode* root = createTree();

    // Get the max depth
    int res = solution.maxDepth(root);

    // Print the result
    cout << "Max Depth: " << res << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0104_Maximum_Depth_of_Binary_Tree.cpp -o test

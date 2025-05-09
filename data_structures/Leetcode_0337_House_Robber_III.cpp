#include <iostream>
#include <algorithm>
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
    pair<int, int> dfs(TreeNode* node) {
        if (node == nullptr) {
            return {0, 0}; // Base case: if the node is nullptr, return {0, 0}
        }

        // Recursively calculate the pair for the left and right subtrees
        pair<int, int> leftPair = dfs(node->left);
        pair<int, int> rightPair = dfs(node->right);

        // 'withRoot' means including the current node and excluding its children
        int withRoot = node->val + leftPair.second + rightPair.second;
        // 'withoutRoot' means excluding the current node, but taking the max from children
        int withoutRoot = max(leftPair.first, leftPair.second) + max(rightPair.first, rightPair.second);

        // Return a pair: the first element is the result including the current node, the second excluding it
        return {withRoot, withoutRoot};
    }

    int rob(TreeNode* root) {
        pair<int, int> result = dfs(root);
        // Return the maximum of robbing this node or not robbing it
        return max(result.first, result.second);
    }
};

// Function to build a binary tree from level-order input
TreeNode* buildTree(const vector<int>& values) {
    if (values.empty()) return nullptr;
    
    TreeNode* root = new TreeNode(values[0]);
    vector<TreeNode*> queue = {root};
    int index = 1;
    
    while (index < values.size()) {
        TreeNode* node = queue.front();
        queue.erase(queue.begin());
        
        if (values[index] != -1) {
            node->left = new TreeNode(values[index]);
            queue.push_back(node->left);
        }
        index++;
        
        if (index < values.size() && values[index] != -1) {
            node->right = new TreeNode(values[index]);
            queue.push_back(node->right);
        }
        index++;
    }
    
    return root;
}

int main() {
    // Test case: [3, 2, 3, -1, 3, -1, 1]
    vector<int> treeValues = {3, 2, 3, -1, 3, -1, 1};
    TreeNode* root = buildTree(treeValues);

    Solution solution;
    cout << "Maximum amount that can be robbed: " << solution.rob(root) << endl;
    return 0;
}

// g++ -std=c++17 Leetcode_0337_House_Robber_III.cpp -o test
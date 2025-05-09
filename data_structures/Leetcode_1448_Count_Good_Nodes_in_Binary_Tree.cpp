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
    int goodNodes(TreeNode* root) {
        return dfs(root, root->val); // Start DFS from the root with its value as the initial max
    }
    
    int dfs(TreeNode* node, int maxVal) {
        if (!node) {
            return 0;
        }

        int res = (node->val >= maxVal) ? 1 : 0;  // Check if the current node is a "good" node
        maxVal = max(maxVal, node->val);  // Update the maximum value encountered so far
        res += dfs(node->left, maxVal);  // Recurse on the left child
        res += dfs(node->right, maxVal);  // Recurse on the right child
        return res;
    }
};

// Helper functions to create and print the tree
TreeNode* createTree() {
    TreeNode* n1 = new TreeNode(3);
    TreeNode* n2 = new TreeNode(1);
    TreeNode* n3 = new TreeNode(4);
    TreeNode* n4 = new TreeNode(3);
    TreeNode* n5 = new TreeNode(1);
    TreeNode* n6 = new TreeNode(5);

    n1->left = n2;
    n1->right = n3;
    n3->left = n4;
    n3->right = n5;
    n4->right = n6;

    return n1;
}

void printGoodNodes(Solution& sol, TreeNode* root) {
    cout << "Good nodes: " << sol.goodNodes(root) << endl;
}

int main() {
    Solution sol;
    
    // Test Case 1
    TreeNode* root1 = createTree();
    printGoodNodes(sol, root1);  // Expected output: 4 (Nodes 3, 4, 3, and 5 are good nodes)

    // Test Case 2: Single node tree
    TreeNode* root2 = new TreeNode(1);
    printGoodNodes(sol, root2);  // Expected output: 1 (Only node 1 is good)

    // Test Case 3: Tree with all nodes having the same value
    TreeNode* n1 = new TreeNode(3);
    TreeNode* n2 = new TreeNode(3);
    TreeNode* n3 = new TreeNode(3);
    TreeNode* n4 = new TreeNode(3);
    TreeNode* n5 = new TreeNode(3);
    TreeNode* n6 = new TreeNode(3);

    n1->left = n2;
    n1->right = n3;
    n2->left = n4;
    n2->right = n5;
    n3->left = n6;
    
    printGoodNodes(sol, n1);  // Expected output: 6 (All nodes are good since their values are equal)

    return 0;
}
// g++ -std=c++17 Leetcode_1448_Count_Good_Nodes_in_Binary_Tree.cpp -o test

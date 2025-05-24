#include <iostream>

using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    TreeNode* flatten(TreeNode* root) {
        return dfs(root);
    }

private:
    // dfs returns the tail (last node) of the flattened subtree
    TreeNode* dfs(TreeNode* root) {
        if (!root)
            return nullptr;

        TreeNode* leftTail = dfs(root->left);
        TreeNode* rightTail = dfs(root->right);

        // If there's a left subtree, attach it to the right
        if (root->left) {
            leftTail->right = root->right;
            root->right = root->left;
            root->left = nullptr;
        }

        // Determine the new tail
        if (rightTail)
            return rightTail;
        else if (leftTail)
            return leftTail;
        else
            return root;
    }
};

// Helper to print the flattened tree
void printFlattenedTree(TreeNode* root) {
    while (root) {
        cout << root->val << " -> ";
        root = root->right;
    }
    cout << "null" << endl;
}

int main() {
    // Build input tree:
    //         1
    //        / \
    //       2   5
    //      / \   \
    //     3  4   6

    TreeNode n6(6);
    TreeNode n5(5, nullptr, &n6);
    TreeNode n4(4);
    TreeNode n3(3);
    TreeNode n2(2, &n3, &n4);
    TreeNode root(1, &n2, &n5);

    Solution sol;
    sol.flatten(&root);

    cout << "Flattened Tree:" << endl;
    printFlattenedTree(&root);

    return 0;
}

// g++ -std=c++17 Leetcode_0114_Flatten_Binary_Tree_to_Linked_List.cpp -o test
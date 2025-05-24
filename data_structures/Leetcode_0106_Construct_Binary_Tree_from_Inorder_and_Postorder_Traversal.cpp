#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
private:
    unordered_map<int, int> index_map; // To store index of values in inorder
    vector<int> postorder;             // Global postorder list

    // Helper function to build the tree recursively
    TreeNode* build(int in_left, int in_right, int post_left, int post_right) {
        if (in_left > in_right || post_left > post_right)
            return nullptr;

        int root_val = postorder[post_right];
        TreeNode* root = new TreeNode(root_val);

        int in_root = index_map[root_val];

        int left_size = in_root - in_left;

        // Recurse to build left and right subtrees
        root->left = build(in_left, in_left + left_size - 1,
                          post_left, post_left + left_size - 1);
        root->right = build(in_root + 1, in_right,
                           post_left + left_size, post_right - 1);

        return root;
    }

public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& po) {
        postorder = po;
        for (int i = 0; i < inorder.size(); ++i) {
            index_map[inorder[i]] = i;
        }
        return build(0, inorder.size() - 1, 0, postorder.size() - 1);
    }
};

// Helper to print tree preorder (for verification)
void print_tree(TreeNode* root) {
    if (!root) return;
    cout << root->val << " ";
    print_tree(root->left);
    print_tree(root->right);
}

// Main test
int main() {
    vector<int> inorder = {9, 3, 15, 20, 7};
    vector<int> postorder = {9, 15, 7, 20, 3};

    Solution sol;
    TreeNode* root = sol.buildTree(inorder, postorder);

    cout << "Reconstructed Tree (Preorder): ";
    print_tree(root); // Expected: 3 9 20 15 7
    cout << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0106_Construct_Binary_Tree_from_Inorder_and_Postorder_Traversal.cpp -o test
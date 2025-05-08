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
    TreeNode* invertTree(TreeNode* root) {
        if (!root) return nullptr;

        queue<TreeNode*> que;
        que.push(root);

        while (!que.empty()) {
            int n = que.size();
            for (int i = 0; i < n; ++i) {
                TreeNode* cur = que.front();
                que.pop();
                // Swap left and right children
                swap(cur->left, cur->right);
                
                // Add left and right children to the queue if they exist
                if (cur->left) que.push(cur->left);
                if (cur->right) que.push(cur->right);
            }
        }

        return root;
    }
};

// Helper function to print the tree (In-order traversal)
void printTree(TreeNode* root) {
    if (!root) return;
    printTree(root->left);
    cout << root->val << " ";
    printTree(root->right);
}

int main() {
    Solution solution;

    // Create the tree nodes
    TreeNode* n1 = new TreeNode(4);
    TreeNode* n2 = new TreeNode(2);
    TreeNode* n3 = new TreeNode(7);
    TreeNode* n4 = new TreeNode(1);
    TreeNode* n5 = new TreeNode(3);
    TreeNode* n6 = new TreeNode(6);
    TreeNode* n7 = new TreeNode(9);

    // Construct the tree
    n1->left = n2;
    n1->right = n3;
    n2->left = n4;
    n2->right = n5;
    n3->left = n6;
    n3->right = n7;

    // Invert the tree
    TreeNode* invertedTree = solution.invertTree(n1);

    // Print the inverted tree using in-order traversal
    cout << "Inverted Tree: ";
    printTree(invertedTree);
    cout << endl;

    return 0;
}
// g++ -std=c++17 Leetcode_0226_Invert_Binary_Tree.cpp -o test

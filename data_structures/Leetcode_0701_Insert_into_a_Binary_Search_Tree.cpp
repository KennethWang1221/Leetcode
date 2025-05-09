#include <iostream>
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
    // Function to insert a new value into the Binary Search Tree (BST)
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if (root == nullptr) {
            return new TreeNode(val);  // If the root is null, create a new node with the value
        }

        TreeNode* cur = root;
        // Traverse the tree to find the correct spot for the new node
        while (cur != nullptr) {
            if (val < cur->val) {
                // If the value is smaller, go to the left subtree
                if (cur->left == nullptr) {
                    cur->left = new TreeNode(val);  // Insert the value here
                    break;
                } else {
                    cur = cur->left;  // Move to the left child
                }
            } else {
                // If the value is greater, go to the right subtree
                if (cur->right == nullptr) {
                    cur->right = new TreeNode(val);  // Insert the value here
                    break;
                } else {
                    cur = cur->right;  // Move to the right child
                }
            }
        }

        return root;  // Return the root of the tree after insertion
    }

    // Function to print the tree in In-order traversal (left, root, right)
    void printTree(TreeNode* root) {
        if (root != nullptr) {
            printTree(root->left);
            cout << root->val << " ";
            printTree(root->right);
        }
    }
};

int main() {
    // Test case 1: Insert value 5 into a given BST
    Solution sol;

    TreeNode* root = new TreeNode(4);
    root->left = new TreeNode(2);
    root->right = new TreeNode(7);
    root->left->left = new TreeNode(1);
    root->left->right = new TreeNode(3);

    cout << "Original BST (In-order): ";
    sol.printTree(root);
    cout << endl;

    // Insert value 5
    root = sol.insertIntoBST(root, 5);
    cout << "After inserting 5 (In-order): ";
    sol.printTree(root);
    cout << endl;

    // Test case 2: Insert value 0 into a given BST
    TreeNode* root2 = new TreeNode(4);
    root2->left = new TreeNode(2);
    root2->right = new TreeNode(7);
    root2->left->left = new TreeNode(1);
    root2->left->right = new TreeNode(3);

    cout << "Original BST (In-order): ";
    sol.printTree(root2);
    cout << endl;

    // Insert value 0
    root2 = sol.insertIntoBST(root2, 0);
    cout << "After inserting 0 (In-order): ";
    sol.printTree(root2);
    cout << endl;

    return 0;
}



// g++ -std=c++17 Leetcode_0701_Insert_into_a_Binary_Search_Tree.cpp -o test

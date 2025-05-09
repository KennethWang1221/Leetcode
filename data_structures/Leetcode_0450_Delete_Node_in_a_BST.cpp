#include <iostream>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        if (!root) return nullptr;  // If the node is empty, return nullptr
        
        if (root->val < key) {
            root->right = deleteNode(root->right, key);  // Recursively move to the right subtree
        } else if (root->val > key) {
            root->left = deleteNode(root->left, key);  // Recursively move to the left subtree
        } else {
            // Case 1: Node has no left child
            if (!root->left) {
                TreeNode* temp = root->right;
                delete root;
                return temp;
            }
            // Case 2: Node has no right child
            if (!root->right) {
                TreeNode* temp = root->left;
                delete root;
                return temp;
            }
            // Case 3: Node has both left and right children
            TreeNode* node = root->right;
            while (node && node->left) {
                node = node->left;  // Find the leftmost node in the right subtree
            }
            root->val = node->val;  // Replace current node value with the leftmost node value
            root->right = deleteNode(root->right, node->val);  // Recursively delete the node in the right subtree
        }
        return root;
    }

    // Helper function to print the tree in-order (left, root, right)
    void printTree(TreeNode* root) {
        if (root != nullptr) {
            printTree(root->left);
            cout << root->val << " ";
            printTree(root->right);
        }
    }
};

int main() {
    Solution solution;

    // Test Case 1: Deleting a node with value 5 (root node)
    TreeNode* n1 = new TreeNode(5);
    TreeNode* n2 = new TreeNode(3);
    TreeNode* n3 = new TreeNode(6);
    TreeNode* n4 = new TreeNode(2);
    TreeNode* n5 = new TreeNode(4);
    TreeNode* n6 = new TreeNode(7);

    n1->left = n2;
    n1->right = n3;
    n2->left = n4;
    n2->right = n5;
    n3->right = n6;

    cout << "Original Tree: ";
    solution.printTree(n1);
    cout << endl;

    // Delete node with value 5
    TreeNode* res = solution.deleteNode(n1, 5);
    cout << "After deleting 5: ";
    solution.printTree(res);
    cout << endl;

    // Test Case 2: Deleting a leaf node with value 7
    n1 = new TreeNode(5);
    n2 = new TreeNode(3);
    n3 = new TreeNode(6);
    n4 = new TreeNode(2);
    n5 = new TreeNode(4);
    n6 = new TreeNode(7);

    n1->left = n2;
    n1->right = n3;
    n2->left = n4;
    n2->right = n5;
    n3->right = n6;

    cout << "Original Tree: ";
    solution.printTree(n1);
    cout << endl;

    // Delete node with value 7 (a leaf node)
    res = solution.deleteNode(n1, 7);
    cout << "After deleting 7: ";
    solution.printTree(res);
    cout << endl;

    // Test Case 3: Deleting a node with value 3 (node with both left and right children)
    n1 = new TreeNode(5);
    n2 = new TreeNode(3);
    n3 = new TreeNode(6);
    n4 = new TreeNode(2);
    n5 = new TreeNode(4);
    n6 = new TreeNode(7);

    n1->left = n2;
    n1->right = n3;
    n2->left = n4;
    n2->right = n5;
    n3->right = n6;

    cout << "Original Tree: ";
    solution.printTree(n1);
    cout << endl;

    // Delete node with value 3 (node with left and right children)
    res = solution.deleteNode(n1, 3);
    cout << "After deleting 3: ";
    solution.printTree(res);
    cout << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0450_Delete_Node_in_a_BST.cpp -o test

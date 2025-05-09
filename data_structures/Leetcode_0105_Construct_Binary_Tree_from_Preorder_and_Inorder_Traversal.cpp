#include <iostream>
#include <vector>
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
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if (preorder.empty() || inorder.empty()) {
            return nullptr;
        }
        
        // The first element in preorder is the root node
        TreeNode* root = new TreeNode(preorder[0]);
        
        // Find the index of the root in inorder
        int mid = find(inorder.begin(), inorder.end(), preorder[0]) - inorder.begin();
        
        // Recursively build the left and right subtrees
        vector<int> leftPreorder(preorder.begin() + 1, preorder.begin() + mid + 1);
        vector<int> leftInorder(inorder.begin(), inorder.begin() + mid);
        vector<int> rightPreorder(preorder.begin() + mid + 1, preorder.end());
        vector<int> rightInorder(inorder.begin() + mid + 1, inorder.end());
        
        root->left = buildTree(leftPreorder, leftInorder);
        root->right = buildTree(rightPreorder, rightInorder);
        
        return root;
    }
};

// Helper function to print the tree in preorder (for validation)
void printTree(TreeNode* root) {
    if (root) {
        cout << root->val << " ";
        printTree(root->left);
        printTree(root->right);
    }
}

int main() {
    Solution sol;

    // Test case 1: Given preorder and inorder for a typical tree
    vector<int> preorder1 = {3, 9, 20, 15, 7};
    vector<int> inorder1 = {9, 3, 15, 20, 7};
    
    // Build tree
    TreeNode* root1 = sol.buildTree(preorder1, inorder1);
    
    // Print the tree in preorder (to validate the structure)
    cout << "Preorder of the tree constructed from test case 1: ";
    printTree(root1);  // Expected output: 3 9 20 15 7
    cout << endl;

    // Test case 2: Edge case where the tree is just one node
    vector<int> preorder2 = {1};
    vector<int> inorder2 = {1};
    
    // Build tree
    TreeNode* root2 = sol.buildTree(preorder2, inorder2);
    
    // Print the tree in preorder (to validate the structure)
    cout << "Preorder of the tree constructed from test case 2: ";
    printTree(root2);  // Expected output: 1
    cout << endl;

    // Test case 3: A left-skewed tree (all nodes on the left)
    vector<int> preorder3 = {5, 4, 3, 2, 1};
    vector<int> inorder3 = {1, 2, 3, 4, 5};
    
    // Build tree
    TreeNode* root3 = sol.buildTree(preorder3, inorder3);
    
    // Print the tree in preorder (to validate the structure)
    cout << "Preorder of the tree constructed from test case 3: ";
    printTree(root3);  // Expected output: 5 4 3 2 1
    cout << endl;

    // Test case 4: A right-skewed tree (all nodes on the right)
    vector<int> preorder4 = {1, 2, 3, 4, 5};
    vector<int> inorder4 = {1, 2, 3, 4, 5};
    
    // Build tree
    TreeNode* root4 = sol.buildTree(preorder4, inorder4);
    
    // Print the tree in preorder (to validate the structure)
    cout << "Preorder of the tree constructed from test case 4: ";
    printTree(root4);  // Expected output: 1 2 3 4 5
    cout << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0105_Construct_Binary_Tree_from_Preorder_and_Inorder_Traversal.cpp -o test

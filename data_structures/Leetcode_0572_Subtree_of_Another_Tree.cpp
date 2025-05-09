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
    // Function to check if two trees are the same
    bool sameTree(TreeNode* s, TreeNode* t) {
        if (!s && !t) return true;  // Both are null, same tree
        if (!s || !t || s->val != t->val) return false;  // One is null or values don't match
        
        // Recursively check left and right subtrees
        return sameTree(s->left, t->left) && sameTree(s->right, t->right);
    }

    // Function to check if t is a subtree of s
    bool isSubtree(TreeNode* s, TreeNode* t) {
        if (!t) return true;  // An empty tree is always a subtree
        if (!s) return false;  // If s is empty, t can't be a subtree

        // Check if the current node matches the subtree
        if (sameTree(s, t)) return true;

        // Otherwise, check the left and right subtrees of s
        return isSubtree(s->left, t) || isSubtree(s->right, t);
    }
};

// Helper function to create the tree
TreeNode* createTree() {
    TreeNode* n1 = new TreeNode(3);
    TreeNode* n2 = new TreeNode(4);
    TreeNode* n3 = new TreeNode(5);
    TreeNode* n4 = new TreeNode(1);
    TreeNode* n5 = new TreeNode(2);

    n1->left = n2;
    n1->right = n3;
    n2->left = n4;
    n2->right = n5;

    return n1;
}

TreeNode* createSubTree() {
    TreeNode* s1 = new TreeNode(4);
    TreeNode* s2 = new TreeNode(1);
    TreeNode* s3 = new TreeNode(2);
    
    s1->left = s2;
    s1->right = s3;
    
    return s1;
}

int main() {
    Solution solution;

    // Create the main tree and the subtree
    TreeNode* root = createTree();
    TreeNode* subRoot = createSubTree();

    // Check if subRoot is a subtree of root
    bool result = solution.isSubtree(root, subRoot);

    cout << "Is subRoot a subtree of root? " << (result ? "Yes" : "No") << endl;

    return 0;
}



// g++ -std=c++17 Leetcode_0572_Subtree_of_Another_Tree.cpp -o test
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
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (!p && !q) {
            return true;  // Both trees are empty
        }
        if (!p || !q || p->val != q->val) {
            return false;  // One tree is empty, or values don't match
        }

        // Recursively check the left and right subtrees
        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }
};

// Helper function to create a tree
TreeNode* createTree1() {
    TreeNode* p1 = new TreeNode(1);
    TreeNode* p2 = new TreeNode(2);
    TreeNode* p3 = new TreeNode(3);

    p1->left = p2;
    p1->right = p3;

    return p1;
}

TreeNode* createTree2() {
    TreeNode* q1 = new TreeNode(1);
    TreeNode* q2 = new TreeNode(2);
    TreeNode* q3 = new TreeNode(3);

    q1->left = q2;
    q1->right = q3;

    return q1;
}

int main() {
    Solution solution;

    // Create trees p and q
    TreeNode* p = createTree1();
    TreeNode* q = createTree2();

    // Check if the trees are the same
    bool result = solution.isSameTree(p, q);

    cout << "Are the two trees the same? " << (result ? "Yes" : "No") << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0100_Same_Tree.cpp -o test

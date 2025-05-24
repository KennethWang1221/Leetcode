#include <iostream>
#include <stack>

using namespace std;

// Definition for binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class BSTIterator {
private:
    stack<TreeNode*> st;

    // Helper to push all left nodes onto the stack
    void pushAllLeft(TreeNode* node) {
        while (node != nullptr) {
            st.push(node);
            node = node->left;
        }
    }

public:
    // Constructor
    BSTIterator(TreeNode* root) {
        pushAllLeft(root);
    }

    // Returns whether we have a next smallest number
    bool hasNext() {
        return !st.empty();
    }

    // Returns the next smallest number
    int next() {
        TreeNode* node = st.top(); st.pop();
        if (node->right) {
            pushAllLeft(node->right);
        }
        return node->val;
    }
};

// Helper function to build a sample BST
TreeNode* buildSampleTree() {
    TreeNode* root = new TreeNode(7);
    root->left = new TreeNode(3);
    root->right = new TreeNode(15);
    root->right->left = new TreeNode(9);
    root->right->right = new TreeNode(20);

    return root;
}

int main() {
    TreeNode* root = buildSampleTree();

    BSTIterator it(root);

    cout << "BST Inorder Traversal via Iterator:\n";
    while (it.hasNext()) {
        cout << it.next() << " ";
    }
    cout << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0173_Binary_Search_Tree_Iterator.cpp -o test
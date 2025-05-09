#include <iostream>
using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// Function to find the Lowest Common Ancestor
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (root == nullptr) {
        return nullptr;
    }
    
    TreeNode* cur = root;
    while (cur) {
        if (p->val > cur->val && q->val > cur->val) {
            cur = cur->right;
        }
        else if (p->val < cur->val && q->val < cur->val) {
            cur = cur->left;
        }
        else {
            return cur;  // LCA found
        }
    }
    return nullptr;
}

// Test the function
int main() {
    // Creating the binary tree
    TreeNode* n1 = new TreeNode(6);
    TreeNode* n2 = new TreeNode(2);
    TreeNode* n3 = new TreeNode(8);
    TreeNode* n4 = new TreeNode(0);
    TreeNode* n5 = new TreeNode(4);
    TreeNode* n6 = new TreeNode(7);
    TreeNode* n7 = new TreeNode(9);
    TreeNode* n8 = new TreeNode(3);
    TreeNode* n9 = new TreeNode(5);

    n1->left = n2;
    n1->right = n3;
    n2->left = n4;
    n2->right = n5;
    n3->left = n6;
    n3->right = n7;
    n5->left = n8;
    n5->right = n9;

    // Test case 1: LCA of nodes 2 and 8
    TreeNode* p = n2;
    TreeNode* q = n7;
    TreeNode* res = lowestCommonAncestor(n1, p, q);
    cout << "LCA of 2 and 8: " << (res ? to_string(res->val) : "None") << endl;  // Expected output: 6

    // Test case 2: LCA of nodes 2 and 4
    p = n2;
    q = n5;
    res = lowestCommonAncestor(n1, p, q);
    cout << "LCA of 2 and 4: " << (res ? to_string(res->val) : "None") << endl;  // Expected output: 2

    // Test case 3: LCA of nodes 5 and 3
    p = n9;
    q = n8;
    res = lowestCommonAncestor(n1, p, q);
    cout << "LCA of 5 and 3: " << (res ? to_string(res->val) : "None") << endl;  // Expected output: 4

    // Test case 4: LCA of nodes 0 and 9
    p = n4;
    q = n7;
    res = lowestCommonAncestor(n1, p, q);
    cout << "LCA of 0 and 9: " << (res ? to_string(res->val) : "None") << endl;  // Expected output: 6

    // Test case 5: LCA of nodes 3 and 5
    p = n8;
    q = n9;
    res = lowestCommonAncestor(n1, p, q);
    cout << "LCA of 3 and 5: " << (res ? to_string(res->val) : "None") << endl;  // Expected output: 4

    return 0;
}


// g++ -std=c++17 Leetcode_0235_Lowest_Common_Ancestor_of_a_Binary_Search_Tree.cpp -o test

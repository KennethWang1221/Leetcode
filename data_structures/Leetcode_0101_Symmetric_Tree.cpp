#include <iostream>
#include <stack>

using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x, TreeNode* l = nullptr, TreeNode* r = nullptr)
        : val(x), left(l), right(r) {}
};

class Solution {
public:
bool isSymmetric(TreeNode* root) {
    if (!root) return true;

    stack<TreeNode*> st;
    st.push(root->left);
    st.push(root->right);

    while (!st.empty()) {
        TreeNode* rightNode = st.top(); st.pop();
        TreeNode* leftNode = st.top(); st.pop();

        if (!leftNode && !rightNode) continue; // Both are null
        if (!leftNode || !rightNode || leftNode->val != rightNode->val) {
            return false;
        }

        // Push mirrored nodes
        st.push(leftNode->left);
        st.push(rightNode->right);
        st.push(leftNode->right);
        st.push(rightNode->left);
    }

    return true;
}
};
// Test Case
int main() {
    // Build the symmetric tree:
    //         1
    //       /   \
    //      2     2
    //     / \   / \
    //    3  4  4  3
    Solution solution;
    
    TreeNode n7(3); TreeNode n6(4);
    TreeNode n5(4); TreeNode n4(3);
    TreeNode n3(2, &n6, &n7);
    TreeNode n2(2, &n4, &n5);
    TreeNode root(1, &n2, &n3);

    cout << "Is Symmetric? " << boolalpha << bool(solution.isSymmetric(&root)) << endl; // Output: true

    return 0;

}

// g++ -std=c++17 Leetcode_0101_Symmetric_Tree.cpp -o test

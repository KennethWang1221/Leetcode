#include <iostream>
#include <stack>
#include <vector>
using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> result;
        stack<TreeNode*> stack;
        TreeNode* cur = root;

        while (cur || !stack.empty()) {
            // Traverse to the leftmost node
            while (cur) {
                stack.push(cur);
                cur = cur->left;
            }
            
            // Process the node
            cur = stack.top();
            stack.pop();
            result.push_back(cur->val);
            
            // Move to the right subtree
            cur = cur->right;
        }
        
        return result;
    }
};

int main() {
    // Create the tree nodes
    TreeNode* n1 = new TreeNode(1);
    TreeNode* n2 = new TreeNode(0);
    TreeNode* n3 = new TreeNode(2);
    TreeNode* n4 = new TreeNode(3);
    
    // Construct the tree
    n1->left = n2;
    n1->right = n3;
    n3->left = n4;

    Solution s;
    vector<int> res = s.inorderTraversal(n1);

    // Print the result
    for (int val : res) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0094_Binary_Tree_Inorder_Traversal.cpp -o test

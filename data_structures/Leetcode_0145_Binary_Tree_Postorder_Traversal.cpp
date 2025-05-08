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
    vector<int> postorderTraversal(TreeNode* root) {
        if (!root) return {};
        
        stack<TreeNode*> stack;
        vector<int> result;
        TreeNode* lastVisited = nullptr;

        while (!stack.empty() || root != nullptr) {
            if (root) {
                stack.push(root);
                root = root->left;  // go left
            } else {
                TreeNode* peekNode = stack.top();
                // if right child is not visited yet
                if (peekNode->right && lastVisited != peekNode->right) {
                    root = peekNode->right;  // move to the right child
                } else {
                    // visit the node
                    result.push_back(peekNode->val);
                    lastVisited = stack.top();
                    stack.pop();
                }
            }
        }
        
        return result;
    }
};

int main() {
    Solution s;

    // Create the tree nodes
    TreeNode* n1 = new TreeNode(1);
    TreeNode* n2 = new TreeNode(0);
    TreeNode* n3 = new TreeNode(2);
    TreeNode* n4 = new TreeNode(3);

    // Construct the tree
    n1->left = n2;
    n1->right = n3;
    n3->left = n4;

    vector<int> res = s.postorderTraversal(n1);

    // Print the result
    for (int val : res) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0145_Binary_Tree_Postorder_Traversal.cpp -o test

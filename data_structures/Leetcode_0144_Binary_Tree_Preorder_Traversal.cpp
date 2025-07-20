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
    vector<int> preorderTraversal(TreeNode* root) {
        if (!root) return {};
        
        vector<int> result;
        stack<TreeNode*> stack;
        stack.push(root);
        
        while (!stack.empty()) {
            TreeNode* node = stack.top();
            stack.pop();
            result.push_back(node->val);
            
            // Push right child first, then left child (to ensure left is processed first)
            if (node->right) stack.push(node->right);
            if (node->left) stack.push(node->left);
        }
        
        return result;
    }


    vector<int> preorderTraversal_DFS(TreeNode* root) {
        
        vector<int> res;
        traversal(root, res);
        return res;
        
    }

    void traversal(TreeNode* root, vector<int>& res){
        if (!root){
            return;
        }
        res.push_back(root->val);
        traversal(root->left, res);
        traversal(root->right, res);
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
    vector<int> res = s.preorderTraversal(n1);
    vector<int> res_dfs = s.preorderTraversal_DFS(n1);
    // Print the result
    for (int val : res) {
        cout << val << " ";
    }
    for (int val : res_dfs) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0144_Binary_Tree_Preorder_Traversal.cpp -o test

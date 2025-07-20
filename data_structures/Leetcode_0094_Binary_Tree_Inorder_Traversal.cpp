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
        if (!root){return {};}

        vector<int> res;
        stack<TreeNode*> stack;
        TreeNode* cur = root;

        while (!stack.empty() || cur){
            if (cur){
                stack.push(cur);
                cur = cur->left;

            } else{
                cur = stack.top();
                stack.pop();
                res.push_back(cur->val);
                cur = cur->right;
            }
        }
        return res;
        
    }



    vector<int> inorderTraversal_DFS(TreeNode* root) {
        

        vector<int> res;
        traversal(root, res);
        return res;
        
    }

    void traversal(TreeNode* root, vector<int>& res){
        if (!root){return;}

        traversal(root->left, res);
        res.push_back(root->val);
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
    vector<int> res = s.inorderTraversal(n1);
    vector<int> res_dfs = s.inorderTraversal_DFS(n1);

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

// g++ -std=c++17 Leetcode_0094_Binary_Tree_Inorder_Traversal.cpp -o test

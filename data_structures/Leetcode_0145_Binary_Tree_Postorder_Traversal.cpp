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
        
        vector<int> result;
        stack<TreeNode*> stack;
        stack.push(root);
        
        while (!stack.empty()) {
            TreeNode* node = stack.top();
            stack.pop();
            result.push_back(node->val);
            
            if (node->left) stack.push(node->left);
            if (node->right) stack.push(node->right);
            
        }
        reverse(result.begin(), result.end());   
        return result;
    }

    vector<int> postorderTraversal_DFS(TreeNode* root) {
        vector<int> res;
        traversal(root, res);
        return res;
        
    }

    void traversal(TreeNode* root, vector<int>& res){
        if (!root){return;}
        traversal(root->left, res);
        traversal(root->right, res);
        res.push_back(root->val);
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
    vector<int> res_dfs = s.postorderTraversal_DFS(n1);

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


// g++ -std=c++17 Leetcode_0145_Binary_Tree_Postorder_Traversal.cpp -o test

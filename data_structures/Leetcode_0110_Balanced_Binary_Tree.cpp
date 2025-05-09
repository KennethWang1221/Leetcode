#include <iostream>
#include <unordered_map>
#include <stack>
#include <algorithm>

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
    bool isBalanced(TreeNode* root) {
        if (!root) {
            return true;
        }

        unordered_map<TreeNode*, int> height_map;
        stack<TreeNode*> node_stack;
        node_stack.push(root);
        
        while (!node_stack.empty()) {
            TreeNode* node = node_stack.top();
            node_stack.pop();
            
            if (node) {
                node_stack.push(node);  // Push the node again to mark it processed later
                node_stack.push(nullptr);  // A marker to know when to process the node

                if (node->left) node_stack.push(node->left);
                if (node->right) node_stack.push(node->right);
            } else {
                TreeNode* real_node = node_stack.top();
                node_stack.pop();
                
                int left_height = height_map.count(real_node->left) ? height_map[real_node->left] : 0;
                int right_height = height_map.count(real_node->right) ? height_map[real_node->right] : 0;

                if (abs(left_height - right_height) > 1) {
                    return false;  // Tree is not balanced
                }
                height_map[real_node] = 1 + max(left_height, right_height);  // Update the height
            }
        }
        return true;  // Tree is balanced
    }
};

// Helper function to create a tree
TreeNode* createTree() {
    TreeNode* n1 = new TreeNode(1);
    TreeNode* n2 = new TreeNode(2);
    TreeNode* n3 = new TreeNode(3);
    TreeNode* n4 = new TreeNode(4);
    TreeNode* n5 = new TreeNode(5);
    TreeNode* n6 = new TreeNode(6);

    n1->left = n2;
    n1->right = n3;
    n2->left = n4;
    n2->right = n5;
    n3->left = n6;

    return n1;
}

int main() {
    Solution solution;

    // Create the binary tree
    TreeNode* root = createTree();

    // Check if the tree is balanced
    bool result = solution.isBalanced(root);

    cout << "Is the tree balanced? " << (result ? "Yes" : "No") << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0543_Diameter_of_Binary_Tree.cpp -o test

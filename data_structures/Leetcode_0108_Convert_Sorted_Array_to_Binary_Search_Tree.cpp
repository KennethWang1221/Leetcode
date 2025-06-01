#include <iostream>
#include <vector>

using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return buildBST(nums, 0, nums.size() - 1);
    }

private:
    TreeNode* buildBST(vector<int>& nums, int left, int right) {
        if (left > right) {
            return nullptr;
        }
        int mid = left + (right - left) / 2;
        TreeNode* root = new TreeNode(nums[mid]);
        root->left = buildBST(nums, left, mid - 1);
        root->right = buildBST(nums, mid + 1, right);
        return root;
    }
};

// Helper function to perform in-order traversal and print node values
void inOrder(TreeNode* root) {
    if (root == nullptr) {
        return;
    }
    inOrder(root->left);
    cout << root->val << " ";
    inOrder(root->right);
}

int main() {
    // Test case
    vector<int> nums = {-10, -3, 0, 5, 9};
    
    Solution solution;
    TreeNode* root = solution.sortedArrayToBST(nums);
    
    cout << "In-order traversal of the constructed BST: ";
    inOrder(root); // Output should be: -10 -3 0 5 9
    cout << endl;
    
    return 0;
}
// g++ -std=c++17 Leetcode_0108_Convert_Sorted_Array_to_Binary_Search_Tree.cpp -o test
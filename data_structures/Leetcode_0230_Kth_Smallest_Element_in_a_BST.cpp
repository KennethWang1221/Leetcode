#include <iostream>
#include <vector>
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
    int kthSmallest(TreeNode* root, int k) {
        vector<int> res;
        if (!root) return -1; // If root is null, return -1

        // Perform level order traversal (BFS)
        vector<TreeNode*> q = {root};
        while (!q.empty()) {
            vector<TreeNode*> level;
            for (TreeNode* cur : q) {
                res.push_back(cur->val);
                if (cur->left) level.push_back(cur->left);
                if (cur->right) level.push_back(cur->right);
            }
            q = level;
        }
        
        // Sort the result vector and return the kth smallest element
        sort(res.begin(), res.end());
        return res[k - 1];
    }
};

// Helper function to print the result
int main() {
    // Test case: Construct the binary tree
    TreeNode* n1 = new TreeNode(3);
    TreeNode* n2 = new TreeNode(1);
    TreeNode* n3 = new TreeNode(4);
    TreeNode* n4 = new TreeNode(2);
    
    n1->left = n2;
    n1->right = n3;
    n2->right = n4;

    // Create an instance of the Solution class and test
    Solution sol;
    int res = sol.kthSmallest(n1, 1);

    cout << "The 1st smallest element is: " << res << endl;  // Expected output: 1

    return 0;
}


// g++ -std=c++17 Leetcode_0230_Kth_Smallest_Element_in_a_BST.cpp -o test

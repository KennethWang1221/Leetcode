#include <iostream>
#include <queue>
#include <vector>
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
    vector<int> rightSideView(TreeNode* root) {
        vector<int> result;
        if (!root) {
            return result;
        }

        queue<TreeNode*> que;
        que.push(root);

        while (!que.empty()) {
            int level_size = que.size();
            // Traverse the current level
            for (int i = 0; i < level_size; ++i) {
                TreeNode* node = que.front();
                que.pop();
                
                // If it is the last node in the level, add it to the result
                if (i == level_size - 1) {
                    result.push_back(node->val);
                }

                // Add the left and right children to the queue for the next level
                if (node->left) {
                    que.push(node->left);
                }
                if (node->right) {
                    que.push(node->right);
                }
            }
        }

        return result;
    }
};

// Helper function to print the result
void printResult(const vector<int>& result) {
    for (int val : result) {
        cout << val << " ";
    }
    cout << endl;
}

int main() {
    // Create the tree: [1, 2, 3, null, 5, null, 4]
    TreeNode* n1 = new TreeNode(1);
    TreeNode* n2 = new TreeNode(2);
    TreeNode* n3 = new TreeNode(3);
    TreeNode* n4 = new TreeNode(5);
    TreeNode* n5 = new TreeNode(4);

    n1->left = n2;
    n1->right = n3;
    n2->right = n4;
    n3->right = n5;

    Solution s;
    vector<int> res = s.rightSideView(n1);

    cout << "Right side view of the binary tree: ";
    printResult(res);

    return 0;
}


// g++ -std=c++17 Leetcode_0199_Binary_Tree_Right_Side_View.cpp -o test

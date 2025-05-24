#include <iostream>
using namespace std;

// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;
    Node* next;

    Node() : val(0), left(nullptr), right(nullptr), next(nullptr) {}
    Node(int _val) : val(_val), left(nullptr), right(nullptr), next(nullptr) {}
    Node(int _val, Node* _left, Node* _right, Node* _next)
        : val(_val), left(_left), right(_right), next(_next) {}
};

Node* get_next_level(Node* level) {
    while (level) {
        if (level->left)
            return level->left;
        if (level->right)
            return level->right;
        level = level->next;
    }
    return nullptr;
}

class Solution {
public:
    Node* connect(Node* root) {
        if (!root) return nullptr;

        Node* curr = root;
        while (curr) {
            // Dummy head for the next level
            Node dummy(0);
            Node* tail = &dummy;

            // Traverse current level
            Node* node = curr;
            while (node) {
                if (node->left) {
                    tail->next = node->left;
                    tail = tail->next;
                }
                if (node->right) {
                    tail->next = node->right;
                    tail = tail->next;
                }
                node = node->next;
            }

            // Move to next level
            curr = dummy.next;
        }

        return root;
    }
};

int main() {
    // Build example tree:
    //         1
    //       /   \
    //      2     3
    //     / \     \
    //    4   5     7

    Node n7(7);
    Node n5(5);
    Node n4(4);
    Node n3(3, nullptr, &n7, nullptr);
    Node n2(2, &n4, &n5, nullptr);
    Node root(1, &n2, &n3, nullptr);

    Solution sol;
    Node* res = sol.connect(&root);

    // Print connections manually
    cout << "After connection:\n";

    Node* level = res;
    while (level) {
        Node* curr = level;
        while (curr) {
            cout << curr->val;
            if (curr->next)
                cout << "->";
            else
                cout << endl;
            curr = curr->next;
        }
        level = get_next_level(level); // Helper to find next level start
    }

    return 0;
}
// g++ -std=c++17 Leetcode_0117_Populating_Next_Right_Pointers_in_Each_Node_II.cpp -o test
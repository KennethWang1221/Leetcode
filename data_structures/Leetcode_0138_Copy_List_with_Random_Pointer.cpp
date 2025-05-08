#include <iostream>
#include <unordered_map>
using namespace std;

// Definition for a linked list with a random pointer.
class ListNode {
public:
    int val;
    ListNode* next;
    ListNode* random;
    ListNode(int x) : val(x), next(nullptr), random(nullptr) {}
};

// Function to create a linked list from a 2D array with next and random pointers
ListNode* make_list(const vector<vector<int>>& arr) {
    unordered_map<int, ListNode*> nodeMap;
    ListNode* head_node = nullptr;
    ListNode* prev_node = nullptr;

    // Create the nodes
    for (const auto& node_info : arr) {
        int val = node_info[0];
        ListNode* new_node = new ListNode(val);
        nodeMap[val] = new_node;

        if (head_node == nullptr) {
            head_node = new_node;
            prev_node = new_node;
        } else {
            prev_node->next = new_node;
            prev_node = new_node;
        }
    }

    // Set random pointers
    for (const auto& node_info : arr) {
        int val = node_info[0];
        ListNode* current_node = nodeMap[val];
        int random_val = node_info[1];
        if (random_val != -1) {
            current_node->random = nodeMap[random_val];
        }
    }

    return head_node;
}

// Function to print the linked list with random pointers
void print_list(ListNode* head) {
    while (head != nullptr) {
        cout << "Value: " << head->val;
        if (head->random) {
            cout << " Random points to: " << head->random->val;
        } else {
            cout << " Random points to: NULL";
        }
        cout << endl;
        head = head->next;
    }
}

// Solution class with copyRandomList function to deep copy the linked list
class Solution {
public:
    ListNode* copyRandomList(ListNode* head) {
        if (!head) return nullptr;

        // Step 1: Create a mapping from old nodes to new nodes
        unordered_map<ListNode*, ListNode*> oldToCopy;

        ListNode* cur = head;
        while (cur) {
            ListNode* copy = new ListNode(cur->val);
            oldToCopy[cur] = copy;
            cur = cur->next;
        }

        // Step 2: Set next and random pointers for the new nodes
        cur = head;
        while (cur) {
            ListNode* copy = oldToCopy[cur];
            copy->next = oldToCopy[cur->next];
            copy->random = oldToCopy[cur->random];
            cur = cur->next;
        }

        // Step 3: Return the deep-copied list's head
        return oldToCopy[head];
    }
};

int main() {
    Solution solution;

    // Test case: Creating a linked list from a 2D array with random pointers
    vector<vector<int>> arr = {{7, -1}, {13, 0}, {11, 4}, {10, 2}, {1, 0}};
    ListNode* head = make_list(arr);

    // Print the original linked list
    cout << "Original list with random pointers:" << endl;
    print_list(head);

    // Copy the linked list
    ListNode* copiedHead = solution.copyRandomList(head);

    // Print the copied linked list
    cout << "Copied list with random pointers:" << endl;
    print_list(copiedHead);

    return 0;
}

// g++ -std=c++17 Leetcode_0138_Copy_List_with_Random_Pointer.cpp -o test

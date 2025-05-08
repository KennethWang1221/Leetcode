#include <iostream>
using namespace std;

// Definition for singly-linked list
class ListNode {
public:
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

// Function to create a linked list from an array
ListNode* make_list(const vector<int>& arr) {
    ListNode* head_node = nullptr;
    ListNode* p_node = nullptr;
    for (int val : arr) {
        ListNode* new_node = new ListNode(val);
        if (head_node == nullptr) {
            head_node = new_node;
            p_node = new_node;
        } else {
            p_node->next = new_node;
            p_node = new_node;
        }
    }
    return head_node;
}

// Function to print the linked list
void print_list(ListNode* head) {
    while (head != nullptr) {
        cout << head->val;
        if (head->next != nullptr) cout << ", ";
        head = head->next;
    }
    cout << endl;
}

class Solution {
public:
    // Function to detect a cycle in a linked list
    bool hasCycle(ListNode* head) {
        ListNode* fast = head;
        ListNode* slow = head;

        while (fast != nullptr && fast->next != nullptr) {
            fast = fast->next->next;
            slow = slow->next;

            if (fast == slow) {
                return true;  // Cycle detected
            }
        }

        return false;  // No cycle
    }

    // Function to create a cycle in the list for testing purposes
    ListNode* create_cycle(ListNode* head, int pos) {
        if (head == nullptr) {
            return nullptr;
        }

        ListNode* cycle_node = nullptr;
        ListNode* current = head;
        int index = 0;

        while (current->next != nullptr) {
            if (index == pos) {
                cycle_node = current;
            }
            current = current->next;
            index++;
        }

        if (cycle_node != nullptr) {
            current->next = cycle_node;  // Create a cycle by linking the last node to the cycle node
        }

        return head;
    }
};

int main() {
    Solution solution;

    // Create a linked list from an array
    vector<int> arr = {3, 2, 0, 4};
    ListNode* head = make_list(arr);

    // Print the original linked list
    cout << "Original list: ";
    print_list(head);

    // Check for cycle (initially no cycle)
    cout << "Has cycle? " << (solution.hasCycle(head) ? "True" : "False") << endl;  // Expected output: False

    // Create a cycle in the list for testing
    head = solution.create_cycle(head, 1);

    // Check for cycle after creating a cycle
    cout << "Has cycle after creation? " << (solution.hasCycle(head) ? "True" : "False") << endl;  // Expected output: True

    return 0;
}


// g++ -std=c++17 Leetcode_0141_LinkedList_cycle.cpp -o test

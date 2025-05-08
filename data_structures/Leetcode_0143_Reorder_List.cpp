#include <iostream>
using namespace std;

// Definition for singly-linked list.
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
        if (head->next != nullptr) cout << " -> ";
        head = head->next;
    }
    cout << endl;
}

class Solution {
public:
    void reorderList(ListNode* head) {
        if (!head || !head->next || !head->next->next) return;

        // Step 1: Find middle using slow and fast pointers
        ListNode* slow = head;
        ListNode* fast = head->next;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }

        // Step 2: Reverse the second half of the list
        ListNode* second = slow->next;
        slow->next = nullptr; // Split the list into two halves
        ListNode* prev = nullptr;
        while (second) {
            ListNode* tmp = second->next;
            second->next = prev;
            prev = second;
            second = tmp;
        }

        // Step 3: Merge the two halves
        ListNode* first = head;
        second = prev; // Reversed second half
        while (second) {
            ListNode* tmp1 = first->next;
            ListNode* tmp2 = second->next;
            first->next = second;
            second->next = tmp1;
            first = tmp1;
            second = tmp2;
        }
    }
};

int main() {
    Solution solution;

    // Test case 1: Create a linked list from an array
    vector<int> arr = {1, 2, 3, 4, 5};
    ListNode* head = make_list(arr);

    // Print the original linked list
    cout << "Original List: ";
    print_list(head);

    // Reorder the linked list
    solution.reorderList(head);

    // Print the reordered linked list
    cout << "Reordered List: ";
    print_list(head);

    return 0;
}
// g++ -std=c++17 Leetcode_0143_Reorder_List.cpp -o test

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

// Solution class to reverse the linked list
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (head == nullptr) return head;

        ListNode* dummy = nullptr;
        ListNode* pre = dummy;
        ListNode* cur = pre;
        ListNode* future = head;

        while (future != nullptr) {
            pre = cur;
            cur = future;
            future = cur->next;  // Move to the next node
            cur->next = pre;  // Reverse the link
        }
        
        return cur;
    }
};

int main() {
    Solution s;

    // Create a linked list from an array
    vector<int> arr = {1, 2, 3, 4, 5};
    ListNode* head = make_list(arr);

    // Print the original linked list
    cout << "Original list: ";
    print_list(head);

    // Reverse the linked list
    head = s.reverseList(head);

    // Print the reversed linked list
    cout << "Reversed list: ";
    print_list(head);

    return 0;
}
// g++ -std=c++17 Leetcode_0206_Reverse_Linked_list.cpp -o test

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
    ListNode* reverseList(ListNode* head, int left, int right) {
        if (head == nullptr) {
            return head;
        }

        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* pre = dummy;

        // Move `pre` to the node just before `left`
        for (int i = 0; i < left - 1; ++i) {
            if (pre == nullptr) return nullptr;
            pre = pre->next;
        }

        ListNode* p = pre;
        ListNode* cur = pre->next;
        ListNode* tail = cur;
        ListNode* future = cur->next;

        // Reverse the sublist from `left` to `right`
        for (int i = 0; i < right - left + 1; ++i) {
            cur->next = pre;  // Reverse the link
            pre = cur;
            cur = future;
            if (cur != nullptr) {
                future = cur->next;
            }
        }

        p->next = pre;
        tail->next = cur;

        return dummy->next;
    }
};

int main() {
    Solution solution;

    // Test case: Creating a linked list from an array
    vector<int> arr = {1, 2, 3, 4, 5};
    ListNode* head = make_list(arr);

    // Print the original linked list
    cout << "Original List: ";
    print_list(head);

    // Reverse the sublist from position 2 to 4
    int left = 2, right = 4;
    head = solution.reverseList(head, left, right);

    // Print the modified linked list
    cout << "Modified List: ";
    print_list(head);

    return 0;
}


// g++ -std=c++17 Leetcode_0092_Reverse_Linked_listII.cpp -o test

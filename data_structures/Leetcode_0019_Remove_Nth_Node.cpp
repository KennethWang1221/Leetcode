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
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        // Create a dummy node to handle edge cases where the head needs to be removed
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* pre = dummy;
        
        // Fast and slow pointer approach
        ListNode* fast = pre;
        ListNode* slow = pre;

        // Move fast pointer n steps ahead
        for (int i = 0; i < n; i++) {
            fast = fast->next;
        }

        // Move both fast and slow pointers until fast reaches the end
        while (fast->next != nullptr) {
            fast = fast->next;
            slow = slow->next;
        }

        // Remove the n-th node from the end
        slow->next = slow->next->next;
        
        // Return the head of the modified list
        return dummy->next;
    }
};

int main() {
    Solution solution;

    // Create a linked list from an array
    vector<int> arr = {1, 2};
    ListNode* head = make_list(arr);

    // Print the original linked list
    cout << "Original List: ";
    print_list(head);

    // Remove the 2nd node from the end
    int n = 2;
    head = solution.removeNthFromEnd(head, n);

    // Print the modified linked list
    cout << "Modified List: ";
    print_list(head);

    return 0;
}

// g++ -std=c++17 Leetcode_0019_Remove_Nth_Node.cpp -o test

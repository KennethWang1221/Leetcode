#include <iostream>
using namespace std;

// Definition for singly-linked list
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

// Helper functions
ListNode* make_list(const int* arr, int n) {
    if (n == 0) return nullptr;
    ListNode* head = new ListNode(arr[0]);
    ListNode* curr = head;
    for (int i = 1; i < n; ++i) {
        curr->next = new ListNode(arr[i]);
        curr = curr->next;
    }
    return head;
}

void print_list(ListNode* head) {
    while (head != nullptr) {
        cout << head->val;
        if (head->next) cout << " -> ";
        head = head->next;
    }
    cout << endl;
}

// LeetCode 61: Rotate List
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if (!head || !head->next || k == 0)
            return head;

        // Step 1: Count length and find tail
        int length = 1;
        ListNode* tail = head;
        while (tail->next) {
            tail = tail->next;
            length++;
        }

        // Step 2: Normalize k
        k %= length;
        if (k == 0) return head;

        // Step 3: Move fast pointer k steps ahead
        ListNode *fast = head, *slow = head;
        for (int i = 0; i < k; ++i) {
            fast = fast->next;
        }

        // Step 4: Move both pointers until fast reaches last node
        while (fast->next) {
            slow = slow->next;
            fast = fast->next;
        }

        // Step 5: Rotate the list
        ListNode* newHead = slow->next;
        slow->next = nullptr;
        fast->next = head;

        return newHead;
    }
};

// Test Case
int main() {
    int arr[] = {1, 2, 3, 4, 5};
    int n = sizeof(arr) / sizeof(arr[0]);

    ListNode* head = make_list(arr, n);
    cout << "Original List: ";
    print_list(head);

    Solution sol;
    ListNode* rotated = sol.rotateRight(head, 2);
    cout << "After Rotating Right by 2: ";
    print_list(rotated);

    return 0;
}

// g++ -std=c++17 Leetcode_0061_Rotate_list.cpp -o test
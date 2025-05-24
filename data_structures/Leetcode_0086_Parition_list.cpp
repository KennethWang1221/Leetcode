#include <iostream>
using namespace std;

// Definition for singly-linked list
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

// Helper function to create a linked list from an array
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

// Helper function to print the linked list
void print_list(ListNode* head) {
    while (head != nullptr) {
        cout << head->val;
        if (head->next != nullptr)
            cout << ",";
        head = head->next;
    }
    cout << endl;
}

class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        // Dummy nodes to hold left and right partitions
        ListNode left_dummy(0);
        ListNode right_dummy(0);

        ListNode* left = &left_dummy;
        ListNode* right = &right_dummy;

        // Traverse input list and partition
        while (head) {
            if (head->val < x) {
                left->next = head;
                left = left->next;
            } else {
                right->next = head;
                right = right->next;
            }
            head = head->next;
        }

        // Join the two partitions
        left->next = right_dummy.next;
        right->next = nullptr;

        return left_dummy.next;
    }
};

int main() {
    Solution sol;
    int arr[] = {1, 4, 3, 2, 5, 2};
    int n = sizeof(arr) / sizeof(arr[0]);

    ListNode* head = make_list(arr, n);
    cout << "Original List: ";
    print_list(head);

    head = sol.partition(head, 3);
    cout << "Partitioned List (x=3): ";
    print_list(head);

    return 0;
}

// g++ -std=c++17 Leetcode_0086_Parition_list.cpp -o test

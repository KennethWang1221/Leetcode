#include <iostream>
#include <vector>

using namespace std;

// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

// Helper functions
ListNode* make_list(const vector<int>& arr) {
    if (arr.empty()) return nullptr;

    ListNode* head = new ListNode(arr[0]);
    ListNode* curr = head;

    for (size_t i = 1; i < arr.size(); ++i) {
        curr->next = new ListNode(arr[i]);
        curr = curr->next;
    }

    return head;
}

void print_list(ListNode* head) {
    while (head != nullptr) {
        cout << head->val;
        if (head->next != nullptr)
            cout << "->";
        head = head->next;
    }
    cout << endl;
}

class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if (!head || !head->next)
            return head;

        ListNode dummy(0);
        dummy.next = head;
        ListNode* pre = &dummy;
        ListNode* cur = head;

        while (cur && cur->next) {
            if (cur->val == cur->next->val) {
                // Skip all duplicates
                while (cur->next && cur->val == cur->next->val) {
                    cur = cur->next;
                }
                // Remove all duplicates
                pre->next = cur->next;
            } else {
                // No duplicate, move forward
                pre = cur;
            }

            cur = pre->next;
        }

        return dummy.next;
    }
};

// Test Case
int main() {
    Solution sol;
    vector<int> arr = {1, 2, 3, 3, 4, 4, 5};
    ListNode* head = make_list(arr);

    cout << "Original List: ";
    print_list(head);

    head = sol.deleteDuplicates(head);

    cout << "After Removing Duplicates: ";
    print_list(head);

    return 0;
}

// g++ -std=c++17 Leetcode_0082_Remove_Duplicates_list_II.cpp -o test

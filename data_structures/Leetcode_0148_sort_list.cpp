#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* sortList(ListNode* head) {
        if (head == nullptr) {
            return head;
        }

        vector<int> values;
        ListNode* current = head;
        while (current != nullptr) {
            values.push_back(current->val);
            current = current->next;
        }

        sort(values.begin(), values.end());

        ListNode* new_head = new ListNode(0);
        ListNode* new_current = new_head;
        for (int val : values) {
            new_current->next = new ListNode(val);
            new_current = new_current->next;
        }

        return new_head->next;
    }
};

// Helper function to create a linked list from a vector of values
ListNode* create_linked_list(vector<int> values) {
    ListNode dummy(0);
    ListNode* current = &dummy;
    for (int val : values) {
        current->next = new ListNode(val);
        current = current->next;
    }
    return dummy.next;
}

// Helper function to print the linked list
void print_linked_list(ListNode* head) {
    ListNode* current = head;
    while (current != nullptr) {
        cout << current->val;
        if (current->next != nullptr) {
            cout << " -> ";
        }
        current = current->next;
    }
    cout << " -> None" << endl;
}

int main() {
    Solution solution;

    // Test case 1
    ListNode* input_list1 = create_linked_list({4, 2, 1, 3});
    ListNode* sorted_head1 = solution.sortList(input_list1);
    cout << "Test case 1: ";
    print_linked_list(sorted_head1);

    // Test case 2
    ListNode* input_list2 = create_linked_list({5, 3, 8, 7, 2});
    ListNode* sorted_head2 = solution.sortList(input_list2);
    cout << "Test case 2: ";
    print_linked_list(sorted_head2);

    // Test case 3
    ListNode* input_list3 = create_linked_list({});
    ListNode* sorted_head3 = solution.sortList(input_list3);
    cout << "Test case 3: ";
    print_linked_list(sorted_head3);

    return 0;
}
// g++ -std=c++17 Leetcode_0148_sort_list.cpp -o test
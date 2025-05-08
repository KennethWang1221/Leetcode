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

// Solution class to merge two sorted linked lists
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(0);  // Create a dummy node to simplify the merge logic
        ListNode* tail = dummy;  // Tail pointer to build the new merged list
        
        // Merge the two lists while both are not empty
        while (l1 != nullptr && l2 != nullptr) {
            if (l1->val < l2->val) {
                tail->next = new ListNode(l1->val);  // Create a new node and append it to the result
                tail = tail->next;
                l1 = l1->next;  // Move l1 pointer
            } else {
                tail->next = new ListNode(l2->val);
                tail = tail->next;
                l2 = l2->next;  // Move l2 pointer
            }
        }

        // Append remaining nodes of l1 if any
        while (l1 != nullptr) {
            tail->next = new ListNode(l1->val);
            tail = tail->next;
            l1 = l1->next;
        }

        // Append remaining nodes of l2 if any
        while (l2 != nullptr) {
            tail->next = new ListNode(l2->val);
            tail = tail->next;
            l2 = l2->next;
        }

        // Return the merged list starting from the next of dummy node
        return dummy->next;
    }
};

int main() {
    Solution s;

    // Create two linked lists
    ListNode* l1 = make_list({1, 2, 4});
    ListNode* l2 = make_list({1, 3, 4});

    // Print the original linked lists
    cout << "List 1: ";
    print_list(l1);
    cout << "List 2: ";
    print_list(l2);

    // Merge the two lists
    ListNode* mergedList = s.mergeTwoLists(l1, l2);
    
    // Print the merged linked list
    cout << "Merged List: ";
    print_list(mergedList);

    return 0;
}


// g++ -std=c++17 Leetcode_0021_Merge_Two_Sorted_Lists.cpp -o test

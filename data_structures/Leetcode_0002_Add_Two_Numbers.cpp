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
    ListNode* add_two_numbers(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(0);  // Dummy node to simplify the logic
        ListNode* pre = dummy;
        
        int carry = 0;
        while (l1 != nullptr || l2 != nullptr || carry > 0) {
            int v1 = (l1 != nullptr) ? l1->val : 0;
            int v2 = (l2 != nullptr) ? l2->val : 0;
            
            int value = v1 + v2 + carry;
            carry = value / 10;
            value = value % 10;
            
            pre->next = new ListNode(value);  // Create a new node with the calculated value
            pre = pre->next;  // Move to the next node
            
            l1 = (l1 != nullptr) ? l1->next : nullptr;
            l2 = (l2 != nullptr) ? l2->next : nullptr;
        }
        
        return dummy->next;  // Return the actual head of the result list (skip dummy)
    }
};

int main() {
    Solution solution;

    // Test case 1: Creating linked lists from arrays
    vector<int> l1 = {2, 4, 3};
    vector<int> l2 = {5, 6, 4};
    
    ListNode* list1 = make_list(l1);
    ListNode* list2 = make_list(l2);
    
    // Print the original linked lists
    cout << "List 1: ";
    print_list(list1);
    
    cout << "List 2: ";
    print_list(list2);

    // Add the two linked lists
    ListNode* result = solution.add_two_numbers(list1, list2);
    
    // Print the result
    cout << "Result: ";
    print_list(result);

    return 0;
}

// g++ -std=c++17 Leetcode_0002_Add_Two_Numbers.cpp -o test

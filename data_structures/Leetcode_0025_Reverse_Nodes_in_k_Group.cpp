#include <iostream>
using namespace std;

// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

// Helper function to create a linked list from a vector
ListNode* make_list(const vector<int>& arr) {
    ListNode* head = nullptr;
    ListNode* p_node = nullptr;

    for (int val : arr) {
        ListNode* new_node = new ListNode(val);
        if (!head) {
            head = new_node;
            p_node = new_node;
        } else {
            p_node->next = new_node;
            p_node = new_node;
        }
    }
    return head;
}

// Helper function to print a linked list
void print_list(ListNode* head) {
    while (head) {
        cout << head->val << ", ";
        head = head->next;
    }
    cout << endl;
}

class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (!head || k == 1) return head;

        // Create a dummy node and initialize the previous node
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* groupPrev = dummy;

        // Function to get the k-th node from the current position
        auto getKth = [](ListNode* curr, int k) {
            while (curr && k > 0) {
                curr = curr->next;
                k--;
            }
            return curr;
        };

        while (true) {
            // Find the k-th node
            ListNode* kth = getKth(groupPrev, k);
            if (!kth) break;  // If less than k nodes left, stop

            ListNode* groupNext = kth->next;
            ListNode* prev = groupNext;
            ListNode* curr = groupPrev->next;

            // Reverse the group of k nodes
            while (curr != groupNext) {
                ListNode* future = curr->next;
                curr->next = prev;
                prev = curr;
                curr = future;
            }

            // Connect the reversed group back to the previous part
            ListNode* temp = groupPrev->next;
            groupPrev->next = kth;
            groupPrev = temp;
        }

        return dummy->next;
    }
};

int main() {
    // Test case 1: Normal case
    Solution solution;
    vector<int> head = {1, 2, 3, 4, 5};
    int k = 2;

    ListNode* headNode = make_list(head);
    cout << "Original List: ";
    print_list(headNode);
    
    headNode = solution.reverseKGroup(headNode, k);
    
    cout << "Reversed List in K Groups: ";
    print_list(headNode);

    return 0;
}

// g++ -std=c++17 Leetcode_0025_Reverse_Nodes_in_k_Group.cpp -o test

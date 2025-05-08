#include <iostream>
#include <vector>
using namespace std;

// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode* next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* next) : val(x), next(next) {}
};

class Solution {
public:
    // Function to merge two sorted lists
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(0);  // Dummy node to avoid special cases
        ListNode* tail = dummy;

        while (l1 && l2) {
            if (l1->val < l2->val) {
                tail->next = l1;
                l1 = l1->next;
            } else {
                tail->next = l2;
                l2 = l2->next;
            }
            tail = tail->next;
        }

        // If there are remaining nodes in either list, append them
        if (l1) tail->next = l1;
        if (l2) tail->next = l2;

        return dummy->next;
    }

    // Function to merge k sorted lists using divide-and-conquer approach
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;

        while (lists.size() > 1) {
            vector<ListNode*> mergedLists;
            
            // Merge lists in pairs
            for (int i = 0; i < lists.size(); i += 2) {
                ListNode* l1 = lists[i];
                ListNode* l2 = (i + 1 < lists.size()) ? lists[i + 1] : nullptr;
                mergedLists.push_back(mergeTwoLists(l1, l2));
            }

            lists = mergedLists;  // Update the lists with merged lists
        }
        
        return lists[0];  // Return the final merged list
    }
};

// Helper function to convert an array to a linked list
ListNode* arrayToList(const vector<int>& arr) {
    ListNode* dummy = new ListNode(0);
    ListNode* current = dummy;
    for (int val : arr) {
        current->next = new ListNode(val);
        current = current->next;
    }
    return dummy->next;
}

// Helper function to print a linked list
void printList(ListNode* head) {
    while (head) {
        cout << head->val << " ";
        head = head->next;
    }
    cout << endl;
}

int main() {
    Solution solution;

    // Test case 1: Normal case with multiple lists
    vector<ListNode*> lists1 = {
        arrayToList({1, 4, 5}),
        arrayToList({1, 3, 4}),
        arrayToList({2, 6})
    };
    ListNode* merged1 = solution.mergeKLists(lists1);
    printList(merged1);  // Expected output: 1 1 2 3 4 4 5 6

    // Test case 2: Empty list of lists
    vector<ListNode*> lists2;
    ListNode* merged2 = solution.mergeKLists(lists2);
    printList(merged2);  // Expected output: (empty)

    // Test case 3: List of empty lists
    vector<ListNode*> lists3 = {nullptr, nullptr, nullptr};
    ListNode* merged3 = solution.mergeKLists(lists3);
    printList(merged3);  // Expected output: (empty)

    // Test case 4: Single list
    vector<ListNode*> lists4 = {arrayToList({1, 2, 3})};
    ListNode* merged4 = solution.mergeKLists(lists4);
    printList(merged4);  // Expected output: 1 2 3

    return 0;
}
// g++ -std=c++17 Leetcode_0023_Merge_K_Sorted_Lists.cpp -o test

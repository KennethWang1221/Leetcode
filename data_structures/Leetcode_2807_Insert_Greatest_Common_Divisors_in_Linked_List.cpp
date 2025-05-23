#include <iostream>
using namespace std;

// Definition for singly-linked list
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    // Euclidean algorithm to compute GCD
    int gcd(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    ListNode* insertGreatestCommonDivisors(ListNode* head) {
        if (!head || !head->next)
            return head;

        ListNode* curr = head;

        while (curr && curr->next) {
            int a = curr->val;
            int b = curr->next->val;
            int g = gcd(a, b);

            ListNode* newNode = new ListNode(g);
            newNode->next = curr->next;
            curr->next = newNode;

            curr = newNode->next; // Move to original next node
        }

        return head;
    }

    // Helper function to create linked list from vector
    ListNode* buildList(vector<int> nums) {
        if (nums.empty()) return nullptr;
        ListNode* head = new ListNode(nums[0]);
        ListNode* curr = head;
        for (int i = 1; i < nums.size(); ++i) {
            curr->next = new ListNode(nums[i]);
            curr = curr->next;
        }
        return head;
    }

    // Helper function to print and delete list
    void printAndDeleteList(ListNode* head) {
        ListNode* curr = head;
        while (curr) {
            cout << curr->val << " ";
            ListNode* tmp = curr;
            curr = curr->next;
            delete tmp;
        }
        cout << endl;
    }
};

// Test Case
int main() {
    Solution sol;

    // Build input list: 18 -> 6 -> 10 -> 3
    ListNode* head = sol.buildList({18, 6, 10, 3});

    // Insert GCDs
    head = sol.insertGreatestCommonDivisors(head);

    // Output result
    sol.printAndDeleteList(head); // Expected: 18 6 6 2 10 1 3

    return 0;
}
// g++ Leetcode_2807_Insert_Greatest_Common_Divisors_in_Linked_List.cpp -o test  && ./test
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

class KthLargest {
private:
    priority_queue<int, vector<int>, greater<int>> minheap; // Min-heap to store the k largest elements
    int k;

public:
    // Constructor to initialize the KthLargest object with k and an initial list of numbers
    KthLargest(int k, vector<int>& nums) {
        this->k = k;
        for (int n : nums) {
            minheap.push(n); // Push element into min-heap
            if (minheap.size() > k) {
                minheap.pop(); // Maintain size k
            }
        }
    }

    // Method to add a new value and return the k-th largest element
    int add(int val) {
        minheap.push(val); // Add new value to the heap
        if (minheap.size() > k) {
            minheap.pop(); // Remove the smallest element
        }
        return minheap.top(); // The k-th largest element is the smallest in the heap
    }
};

int main() {
    // Test the KthLargest class
    vector<int> nums = {4, 5, 8, 2};
    KthLargest kthLargest(3, nums);
    
    cout << kthLargest.add(3) << endl; // Return 4
    cout << kthLargest.add(5) << endl; // Return 5
    cout << kthLargest.add(10) << endl; // Return 5
    cout << kthLargest.add(9) << endl; // Return 8
    cout << kthLargest.add(4) << endl; // Return 8

    return 0;
}


// g++ -std=c++17 Leetcode_0703_Kth_Largest_Element_in_a_Stream.cpp -o test
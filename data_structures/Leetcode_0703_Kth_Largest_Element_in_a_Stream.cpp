#include <iostream>
#include <vector>
#include <queue>
using namespace std;

class KthLargest {
private:
    priority_queue<int, vector<int>, greater<int>> minheap; // Min-heap to store the k largest elements, By default, it implements a max-heap , meaning the largest element is always at the top. To make it a min-heap , we change the comparison function to greater<int>. By default, priority_queue uses less<int> (i.e., max-heap behavior), where the largest element is at the top. If we want a min-heap , where the smallest element is at the top, we replace the comparator with greater<int>. It tells the priority_queue to order elements in ascending order.  The less<int> comparator defines descending order . It tells the priority_queue to keep the largest element at the top , making it behave like a max-heap . priority_queue<int, vector<int>, greater<int>>: 1. The type of data it stores → int 2. The container it uses to store them → vector<int> 3. The comparison function to order elements → greater<int> (for min-heap)
    int k;

public:
    // Constructor to initialize the KthLargest object with k and an initial list of numbers
    // Using member initializer list for better efficiency. Initialize the member variable k with the value of the parameter k
    KthLargest(int k, vector<int>& nums) : k(k) {
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
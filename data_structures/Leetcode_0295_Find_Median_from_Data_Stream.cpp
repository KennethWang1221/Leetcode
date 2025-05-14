#include <iostream>
#include <queue>
#include <vector>
#include <functional>
using namespace std;

class MedianFinder {
private:
    priority_queue<int> small;  // Max-heap (store the smaller half of numbers)
    priority_queue<int, vector<int>, greater<int>> large;  // Min-heap (store the larger half of numbers)

public:
    /** initialize your data structure here. */
    MedianFinder() {}

    /** Adds a number to the data structure. */
    void addNum(int num) {
        if (large.empty() || num > large.top()) {
            large.push(num);  // Push to the min-heap if it's larger than the smallest element in large
        } else {
            small.push(num);  // Otherwise, push to the max-heap
        }

        // Balance the heaps if the size difference becomes greater than 1
        if (small.size() > large.size() + 1) {
            int val = small.top();
            small.pop();
            large.push(val);  // Move the largest element of small to large
        }
        if (large.size() > small.size() + 1) {
            int val = large.top();
            large.pop();
            small.push(val);  // Move the smallest element of large to small
        }
    }

    /** Returns the median of the current data stream. */
    double findMedian() {
        if (small.size() > large.size()) {
            return small.top();  // If small has more elements, the median is the top of small
        } else if (large.size() > small.size()) {
            return large.top();  // If large has more elements, the median is the top of large
        } else {
            return (small.top() + large.top()) / 2.0;  // If both heaps are of equal size, return the average of the tops
        }
    }
};

int main() {
    MedianFinder medianFinder;
    medianFinder.addNum(1);  // arr = [1]
    medianFinder.addNum(2);  // arr = [1, 2]
    cout << medianFinder.findMedian() << endl;  // Output: 1.5 (i.e., (1 + 2) / 2)

    medianFinder.addNum(3);  // arr = [1, 2, 3]
    cout << medianFinder.findMedian() << endl;  // Output: 2.0

    return 0;
}

// g++ -std=c++17 Leetcode_0295_Find_Median_from_Data_Stream.cpp -o test
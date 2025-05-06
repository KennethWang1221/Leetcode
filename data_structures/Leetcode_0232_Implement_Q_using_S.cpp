#include <iostream>
#include <queue>
using namespace std;

class MyQueue {
public:
    MyQueue() {}

    void push(int x) {
        queue.push(x);
    }

    int pop() {
        int val = queue.front();
        queue.pop();
        return val;
    }

    int peek() {
        return queue.front();
    }

    bool empty() {
        return queue.empty();
    }

private:
    queue<int> queue;  // Using the C++ built-in queue
};

int main() {
    MyQueue queue;
    queue.push(1);
    queue.push(2);
    
    cout << "Peek element: " << queue.peek() << endl;  // Output should be 1
    cout << "Pop element: " << queue.pop() << endl;    // Output should be 1
    cout << "Is queue empty? " << (queue.empty() ? "True" : "False") << endl;  // Output should be False

    return 0;
}
// g++ -std=c++17 Leetcode_0232_Implement_Q_using_S.cpp -o test
#include <iostream>
#include <stack>       // Include the correct header for stack
#include <algorithm>   // For min function
using namespace std;

class MinStack {
public:
    MinStack() {}

    void push(int val) {
        stack.push(val);  // Add to the main stack
        if (minStack.empty()) {
            minStack.push(val);  // Add to the min stack if it's empty
        } else {
            int currentMin = min(val, minStack.top());
            minStack.push(currentMin);  // Keep track of the minimum value
        }
    }

    void pop() {
        stack.pop();       // Remove from the main stack
        minStack.pop();    // Remove from the min stack
    }

    int top() {
        return stack.top();  // Return the top value from the main stack
    }

    int getMin() {
        return minStack.top();  // Return the top value from the min stack (which holds the minimum value)
    }

private:
    std::stack<int> stack;       // Regular stack to store values
    std::stack<int> minStack;    // Stack to store the minimum values
};

int main() {
    MinStack obj;
    obj.push(-2);
    obj.push(0);
    obj.push(-3);

    obj.pop();
    cout << "Top element: " << obj.top() << endl;       // Output should be 0
    cout << "Minimum element: " << obj.getMin() << endl; // Output should be -2

    return 0;
}



// g++ -std=c++17 Leetcode_0155_Min_Stack.cpp -o test
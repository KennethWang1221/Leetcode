#include <iostream>
#include <stack>
using namespace std;

class MyStack {
public:
    MyStack() {}

    void push(int x) {
        stack.push(x);
    }

    int pop() {
        int val = stack.top();
        stack.pop();
        return val;
    }

    int top() {
        return stack.top();
    }

    bool empty() {
        return stack.empty();
    }

private:
    stack<int> stack;  // Using the C++ built-in stack
};

int main() {
    MyStack stack;
    stack.push(1);
    stack.push(2);
    
    cout << "Top element: " << stack.top() << endl;  // Output should be 2
    cout << "Pop element: " << stack.pop() << endl;  // Output should be 2
    cout << "Is stack empty? " << (stack.empty() ? "True" : "False") << endl;  // Output should be False

    return 0;
}
// g++ -std=c++17 Leetcode_0225_Implement_S_using_Q.cpp -o test